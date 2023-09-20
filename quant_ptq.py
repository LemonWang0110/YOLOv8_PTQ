import argparse
import os
import sys
from pathlib import Path
import warnings
import yaml
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.checks import check_imgsz, check_yaml
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import colorstr
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import select_device
from ultralytics.data import build_dataloader

import quant_utils as quant


import re
def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in ('.yaml', '.yml'), f'Attempting to load non-YAML file {file} with yaml_load()'
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data['yaml_file'] = str(file)
        return data

def collect_stats(model, data_loader, num_calib_batch, device):
    # Enable calibrators
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, quant.quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, batch in tqdm(enumerate(data_loader), total=num_calib_batch):
        image = batch['img']
        image = image.to(device, non_blocking=True)
        image = image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        
        model(image)
        if i >= num_calib_batch:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant.quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant.quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, quant.calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")


def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir, device):
    """
        Feed data to the network and calibrate
        Arguments:
            model: detection model
            model_name: name to use when creating state files
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
            calibrator: type of calibration to use (max/histogram)
            hist_percentile: percentiles to be used for historgram calibration
            out_dir: dir to save state files in
    """

    if num_calib_batch > 0:
        print("Calibrating model:")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch, device)

        if not calibrator == "histogram":
            compute_amax(model, method="max")
            calib_output = os.path.join(out_dir, F"{model_name}-max-{num_calib_batch * data_loader.batch_size}.pth")
            torch.save(model.state_dict(), calib_output)
        else:
            for percentile in hist_percentile:
                print(F"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                calib_output = os.path.join(out_dir, F"{model_name}-percentile-{percentile}-{num_calib_batch * data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)

            for method in ["mse", "entropy"]:
                print(F"{method} calibration")
                compute_amax(model, method=method)
                calib_output = os.path.join(out_dir, F"{model_name}-{method}-{num_calib_batch * data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)

def load_model(weight, device) -> DetectionModel:
    model = torch.load(weight, map_location=device)['model']
    model.float()
    model.eval()
    with torch.no_grad():
        model.fuse()
    return model

def prepare_model(calibrator, opt, device):

    # 获取量化校准数据
    data_dict = check_det_dataset(opt.data)
    calib_path = data_dict['val']
    
    # 加载FP32的Pytorch模型
    model = load_model(opt.weights, device)

    quant.initialize_calib_method(per_channel_quantization=True, calib_method=calibrator)  
    quant.replace_to_quantization_module(model, ignore_policy=opt.sensitive_layer)
    model.eval()
    model.cuda()
    
    # Check imgsz
    gs = max(int(model.stride.max() if hasattr(model, 'stride') else 32), 32)  # grid size (max stride)
    imgsz = check_imgsz(opt.imgsz, stride=gs, floor=gs, max_dim=1)
    
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG
    args = get_cfg(cfg=DEFAULT_CFG, overrides=None)

    dataset = YOLODataset(img_path=calib_path,
                          imgsz= imgsz,
                          batch_size=opt.batch_size,
                          augment=False,
                          hyp=args,
                          rect=True,
                          cache=opt.cache,
                          stride=gs,
                          pad=0.5,
                          prefix=colorstr('calib: '))

    # Calib dataloader
    calib_loader = build_dataloader(dataset, opt.batch_size, opt.workers * 2, shuffle=False, rank=-1)  # return dataloader

    return model, calib_loader


def export_onnx(model, onnx_filename, batch_onnx, dynamic_shape, simplify, imgsz=640, prefix=colorstr('calib: ')):
 
    # Update model
    from copy import deepcopy
    model = deepcopy(model).to('cuda')
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for m in model.modules():
        if isinstance(m, (Detect)):
            m.export = True
            m.format = 'onnx'

    # We have to shift to pytorch's fake quant ops before exporting the model to ONNX
    quant.quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Export ONNX for multiple batch sizes
    print("Creating ONNX file: " + onnx_filename)
    dummy_input = torch.randn(batch_onnx, 3, imgsz, imgsz).to('cuda')  

    try:
        import onnx
        with torch.no_grad():
            torch.onnx.export(model, 
                            dummy_input, 
                            onnx_filename, 
                            verbose=False, 
                            opset_version=13, 
                            input_names=['images'],
                            output_names=['output'],
                            dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'}} if dynamic_shape else None,
                            enable_onnx_checker=False, 
                            do_constant_folding=True)

        print('ONNX export success, saved as %s' % onnx_filename)

    except ValueError:
        warnings.warn(UserWarning("Per-channel quantization is not yet supported in Pytorch/ONNX RT (requires ONNX opset 13)"))
        print("Failed to export to ONNX")
        return False

    except Exception as e:
            print(f'{prefix} export failure: {e}')
    
    # Checks
    model_onnx = onnx.load(onnx_filename)  # load onnx model
    onnx.checker.check_model(model_onnx)   # check onnx model
    
    # Simplify
    if simplify:
        try:
            import onnxsim
            print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic_shape,
                    input_shapes={'images': list(dummy_input.shape)} if dynamic_shape else None)

            assert check, 'assert check failed'
            onnx.save(model_onnx, onnx_filename)
        except Exception as e:
            print(f'{prefix} simplifier failure: {e}')

        print(f'{prefix} export success, saved as {onnx_filename} ({file_size(onnx_filename):.1f} MB)')
        # print(f"{prefix} Run ONNX model inference with: 'python detect.py --weights {onnx_filename}'")
        
    # Restore the PSX/TensorRT's fake quant mechanism
    quant.quant_nn.TensorQuantizer.use_fb_fake_quant = False
    # Restore the model to train/test mode, use Detect() layer grid
    model.export = False

    return True

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'ultralytics/cfg/datasets/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov8s.pt', help='model.pt path(s)')
    parser.add_argument('--model-name', '-m', default='yolov8s', help='model name: default yolov8s')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')

    # setting for calibration
    parser.add_argument('--sensitive-layer', default=['model.22.dfl.conv',
                                                      'model.2.cv1.conv'], help='skip sensitive layer: default detect head and second layer')
    parser.add_argument('--num-calib-batch', default=64, type=int, help='Number of batches for calibration. 0 will disable calibration. (default: 4)')
    parser.add_argument('--calibrator', type=str, choices=["max", "histogram"], default="max")
    parser.add_argument('--percentile', nargs='+', type=float, default=[99.9, 99.99, 99.999, 99.9999])
    parser.add_argument('--dynamic', default=False, help='dynamic ONNX axes')
    parser.add_argument('--simplify', default=True, help='simplify ONNX file')
    parser.add_argument('--out-dir', '-o', default=ROOT / 'weights/', help='output folder: default ./runs/finetune')
    parser.add_argument('--batch-size-onnx', type=int, default=1, help='batch size for onnx: default 1')
    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    return opt

def evaluate_accuracy(model, opt, testloader):

    args = dict(mode='val', model=model, data=opt.data, imgsz=opt.imgsz, batch=opt.batch_size, project=opt.out_dir)
    validator = DetectionValidator(args=args)
    info_dict = validator()
    map50 = info_dict['metrics/mAP50(B)']
    map5090 = info_dict['metrics/mAP50-95(B)']
    
    return map50, map5090

def sensitive_analysis(model, opt, data_loader, summary_file='./summary_sensitive_analysis.json'):
    summary = quant.SummaryTool(summary_file)
    # model是插入Q、DQ节点后的模型
    
    print("\033[1;31m所有节点均做量化的评估结果:\033[0m")
    map50_calibrated, map5090_calibrated = evaluate_accuracy(model, opt, data_loader)
    print(f"\033[1;31mCalibration evaluation: mAP@IoU=0.50:{map50_calibrated:.5f}, mAP@IoU=0.50:0.95:{map5090_calibrated:.5f}\033[0m") 
    summary.append([map50_calibrated, map5090_calibrated, "PTQ"])

    print("="*50)
    print("\033[1;31mSensitive Analysis by each layer...\033[0m")
    for i in range(0, len(model.model)):
        layer = model.model[i]
        if quant.have_quantizer(layer):
            print(f"\033[0;32mQuantization disable model.{i}\033[0m")
            quant.disable_quantization(layer).apply()
            map50_calibrated, map5090_calibrated = evaluate_accuracy(model, opt, data_loader)
            
            summary.append([map50_calibrated, map5090_calibrated, f"model.{i}"])
            quant.enable_quantization(layer).apply()
        else:
            print(f"\033[1;33mignore model.{i} because it is {type(layer)}\033[0m")
    
    summary = sorted(summary.data, key=lambda x:x[0], reverse=True)
    print("="*50)
    print("\033[1;31mSensitive summary:\033[0m")
    for n, (map5090_calibrated, map50_calibrated, name) in enumerate(summary[:10]):
        print(f"\033[0;32mTop{n}: Using int8 {name}, map_calibrated = {map5090_calibrated:.5f}\033[0m")


if __name__ == "__main__":

    # 参数
    opt = parse_opt()
    # 设备
    device = select_device(opt.device, opt.batch_size)
    # 准备模型和dataloader
    model, data_loader = prepare_model(calibrator=opt.calibrator, opt=opt, device=device)

    # 校准
    with torch.no_grad():
        calibrate_model(
            model=model,
            model_name=opt.model_name,
            data_loader=data_loader,
            num_calib_batch=opt.num_calib_batch,
            calibrator=opt.calibrator,
            hist_percentile=opt.percentile,
            out_dir=opt.out_dir,
            device=device)

    print("="*80)
    onnx_filename = './weights/yolov8s_ptq.onnx'
    export_onnx(model, onnx_filename, opt.batch_size_onnx, opt.dynamic, opt.simplify, opt.imgsz)
    
    print("="*80)
    # 量化后模型精度
    with torch.no_grad():
        map50_calibrated, map5090_calibrated = evaluate_accuracy(model, opt, data_loader)
        print(f"\033[1;31mCalibration evaluation: mAP@IoU=0.50:{map50_calibrated:.5f}, mAP@IoU=0.50:0.95:{map5090_calibrated:.5f}\033[0m")
    
    # 原始模型精度
    with torch.no_grad():
        with quant.disable_quantization(model):
            map50_Orgin, map5090_Orgin = evaluate_accuracy(model, opt, data_loader)
            print(f"\033[1;31mOrgin evaluation: mAP@IoU=0.50:{map50_Orgin:.5f}, mAP@IoU=0.50:0.95:{map5090_Orgin:.5f}\033[0m")
