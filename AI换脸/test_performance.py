"""
性能测试脚本 - 用于论文第四章实验数据采集
测试人脸替换系统各模块的性能指标
"""
import time
import cv2
import numpy as np
from dofaker import FaceSwapper
import os

def test_detection_speed(image_path, iterations=10):
    """测试人脸检测速度"""
    from dofaker.face_det import FaceAnalysis
    
    det_model = FaceAnalysis(name='buffalo_l', root='weights/models')
    det_model.prepare(ctx_id=0, det_size=(640, 640))
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 预热
    _ = det_model.get(img_rgb)
    
    # 正式测试
    times = []
    for _ in range(iterations):
        start = time.time()
        faces = det_model.get(img_rgb)
        end = time.time()
        times.append((end - start) * 1000)  # 转换为ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"人脸检测性能测试:")
    print(f"  图像尺寸: {img.shape[1]}×{img.shape[0]}")
    print(f"  检测到人脸数: {len(faces)}")
    print(f"  平均检测时间: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  最快: {min(times):.2f} ms")
    print(f"  最慢: {max(times):.2f} ms")
    
    return avg_time, len(faces)

def test_swap_speed(src_face_path, target_path, iterations=5):
    """测试人脸替换速度"""
    print(f"\n人脸替换性能测试:")
    
    # 基础替换（无增强和超分）
    faker_basic = FaceSwapper(use_enhancer=False, use_sr=False)
    
    times = []
    for i in range(iterations):
        start = time.time()
        output = faker_basic.run(target_path, None, src_face_path, 
                                 output_dir='output/test_perf')
        end = time.time()
        times.append((end - start) * 1000)
        print(f"  第{i+1}次: {times[-1]:.2f} ms")
    
    avg_time = np.mean(times)
    print(f"  平均替换时间（基础）: {avg_time:.2f} ± {np.std(times):.2f} ms")
    
    return avg_time

def test_enhance_speed(src_face_path, target_path, iterations=5):
    """测试面部增强速度"""
    print(f"\n面部增强性能测试:")
    
    faker_enhance = FaceSwapper(use_enhancer=True, use_sr=False)
    
    times = []
    for i in range(iterations):
        start = time.time()
        output = faker_enhance.run(target_path, None, src_face_path,
                                   output_dir='output/test_perf')
        end = time.time()
        times.append((end - start) * 1000)
        print(f"  第{i+1}次: {times[-1]:.2f} ms")
    
    avg_time = np.mean(times)
    print(f"  平均时间（替换+增强）: {avg_time:.2f} ± {np.std(times):.2f} ms")
    
    return avg_time

def test_sr_speed(src_face_path, target_path, scale=2, iterations=3):
    """测试超分辨率速度"""
    print(f"\n超分辨率性能测试 (scale={scale}):")
    
    faker_sr = FaceSwapper(use_enhancer=False, use_sr=True, scale=scale)
    
    times = []
    for i in range(iterations):
        start = time.time()
        output = faker_sr.run(target_path, None, src_face_path,
                             output_dir='output/test_perf')
        end = time.time()
        times.append((end - start) * 1000)
        print(f"  第{i+1}次: {times[-1]:.2f} ms")
    
    avg_time = np.mean(times)
    print(f"  平均时间（替换+{scale}x超分）: {avg_time:.2f} ± {np.std(times):.2f} ms")
    
    return avg_time

def test_full_pipeline(src_face_path, target_path, iterations=3):
    """测试完整流程"""
    print(f"\n完整流程性能测试:")
    
    faker_full = FaceSwapper(use_enhancer=True, use_sr=True, scale=2)
    
    times = []
    for i in range(iterations):
        start = time.time()
        output = faker_full.run(target_path, None, src_face_path,
                               output_dir='output/test_perf')
        end = time.time()
        times.append((end - start) * 1000)
        print(f"  第{i+1}次: {times[-1]:.2f} ms")
    
    avg_time = np.mean(times)
    print(f"  平均时间（完整流程）: {avg_time:.2f} ± {np.std(times):.2f} ms")
    
    return avg_time

def get_image_info(image_path):
    """获取图像信息"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return {
        'path': image_path,
        'width': img.shape[1],
        'height': img.shape[0],
        'channels': img.shape[2] if len(img.shape) > 2 else 1,
        'size_mb': os.path.getsize(image_path) / (1024 * 1024)
    }

if __name__ == '__main__':
    print("=" * 60)
    print("人脸替换系统性能测试")
    print("=" * 60)
    
    # 检查测试图像
    test_images = []
    test_dir = 'docs/test'
    
    if os.path.exists(test_dir):
        for f in os.listdir(test_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(test_dir, f))
    
    if len(test_images) < 2:
        print("错误: 需要至少2张测试图像在 docs/test/ 目录")
        print("请添加测试图像后重新运行")
        exit(1)
    
    src_face = test_images[0]
    target_img = test_images[1] if len(test_images) > 1 else test_images[0]
    
    print(f"\n测试图像:")
    src_info = get_image_info(src_face)
    target_info = get_image_info(target_img)
    print(f"  源人脸: {src_info['width']}×{src_info['height']}, {src_info['size_mb']:.2f}MB")
    print(f"  目标图像: {target_info['width']}×{target_info['height']}, {target_info['size_mb']:.2f}MB")
    
    print("\n" + "=" * 60)
    
    # 测试各个模块
    try:
        det_time, num_faces = test_detection_speed(target_img, iterations=10)
        swap_time = test_swap_speed(src_face, target_img, iterations=5)
        enhance_time = test_enhance_speed(src_face, target_img, iterations=5)
        sr_time = test_sr_speed(src_face, target_img, scale=2, iterations=3)
        full_time = test_full_pipeline(src_face, target_img, iterations=3)
        
        print("\n" + "=" * 60)
        print("性能测试总结:")
        print("=" * 60)
        print(f"人脸检测: {det_time:.1f} ms")
        print(f"基础替换: {swap_time:.1f} ms")
        print(f"替换+增强: {enhance_time:.1f} ms")
        print(f"替换+2x超分: {sr_time:.1f} ms")
        print(f"完整流程: {full_time:.1f} ms")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
