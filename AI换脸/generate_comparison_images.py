"""
生成不同配置的换脸效果对比图
用于论文展示实验结果
"""

import cv2
import os
import numpy as np
from dofaker import FaceSwapper
import time
import gc

def add_text_label(img, text, position='top'):
    """在图像上添加文字标签"""
    img_copy = img.copy()
    h, w = img_copy.shape[:2]
    
    # 创建标签背景
    if position == 'top':
        cv2.rectangle(img_copy, (0, 0), (w, 40), (0, 0, 0), -1)
        text_position = (10, 28)
    else:
        cv2.rectangle(img_copy, (0, h-40), (w, h), (0, 0, 0), -1)
        text_position = (10, h-12)
    
    # 添加文字
    cv2.putText(img_copy, text, text_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return img_copy

def create_comparison_grid(images, labels, output_path, cols=3):
    """创建对比网格图"""
    n = len(images)
    rows = (n + cols - 1) // cols
    
    # 统一所有图像大小
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    
    resized_images = []
    for img, label in zip(images, labels):
        # 调整大小
        img_resized = cv2.resize(img, (max_w, max_h))
        # 添加标签
        img_labeled = add_text_label(img_resized, label, 'top')
        resized_images.append(img_labeled)
    
    # 填充空白
    while len(resized_images) < rows * cols:
        blank = np.ones((max_h, max_w, 3), dtype=np.uint8) * 255
        resized_images.append(blank)
    
    # 创建网格
    grid_rows = []
    for i in range(rows):
        row_images = resized_images[i*cols:(i+1)*cols]
        row = np.hstack(row_images)
        grid_rows.append(row)
    
    grid = np.vstack(grid_rows)
    cv2.imwrite(output_path, grid)
    print(f"✓ 保存对比图: {output_path}")
    return grid

def main():
    print("=" * 60)
    print("生成论文实验效果对比图")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = "output/paper_figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试图像路径
    source_path = "docs/test/condition.jpg"  # 源人脸
    target_path = "docs/test/taitan.jpeg"  # 目标图像
    
    if not os.path.exists(source_path) or not os.path.exists(target_path):
        print(f"错误: 测试图像不存在")
        return
    
    # 读取原图
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    
    print(f"\n源图像尺寸: {source_img.shape[:2]}")
    print(f"目标图像尺寸: {target_img.shape[:2]}")
    
    # ========================================
    # 实验1: 不同配置的换脸效果对比
    # ========================================
    print("\n[实验1] 生成不同配置的换脸效果...")
    
    configs = [
        {'enhance': False, 'scale': 1, 'label': '1.原始目标图像'},
        {'enhance': False, 'scale': 1, 'label': '2.基础替换'},
        {'enhance': True, 'scale': 1, 'label': '3.替换+增强'},
        {'enhance': False, 'scale': 2, 'label': '4.替换+2x超分'},
        {'enhance': True, 'scale': 2, 'label': '5.完整流程'},
    ]
    
    images = [target_img]
    labels = ['1.原始目标图像']
    
    for config in configs[1:]:
        print(f"  处理配置: {config['label']}...")
        swapper = FaceSwapper(
            face_det_model='buffalo_l',
            use_enhancer=config['enhance'],
            use_sr=(config['scale'] > 1),
            scale=config['scale']
        )
        
        start = time.time()
        result_path = swapper.run(target_path, dst_face_paths=None, src_face_paths=source_path, output_dir=output_dir)
        elapsed = time.time() - start
        
        # 读取结果图像
        result = cv2.imread(result_path)
        
        images.append(result)
        labels.append(f"{config['label']} ({elapsed*1000:.0f}ms)")
        
        # 释放内存
        del swapper
        gc.collect()
    
    # 创建对比网格
    create_comparison_grid(images, labels, f"{output_dir}/config_comparison.png", cols=3)
    
    # ========================================
    # 实验2: 模块消融实验对比
    # ========================================
    print("\n[实验2] 生成模块消融对比...")
    
    ablation_configs = [
        {'enhance': False, 'scale': 1, 'label': '仅替换 (基准)'},
        {'enhance': True, 'scale': 1, 'label': '+GFPGAN增强'},
        {'enhance': False, 'scale': 2, 'label': '+BSRGAN超分'},
        {'enhance': True, 'scale': 2, 'label': '完整流程'},
    ]
    
    ablation_images = []
    ablation_labels = []
    
    for config in ablation_configs:
        print(f"  处理: {config['label']}...")
        swapper = FaceSwapper(
            face_det_model='buffalo_l',
            use_enhancer=config['enhance'],
            use_sr=(config['scale'] > 1),
            scale=config['scale']
        )
        
        result_path = swapper.run(target_path, dst_face_paths=None, src_face_paths=source_path, output_dir=output_dir)
        result = cv2.imread(result_path)
        ablation_images.append(result)
        ablation_labels.append(config['label'])
    
    create_comparison_grid(
        ablation_images, 
        ablation_labels, 
        f"{output_dir}/ablation_comparison.png", 
        cols=2
    )
    
    # ========================================
    # 实验3: 处理步骤展示
    # ========================================
    print("\n[实验3] 生成处理步骤展示...")
    
    swapper = FaceSwapper(face_det_model='buffalo_l', use_enhancer=False, use_sr=False)
    
    # 检测人脸
    target_faces = swapper.det_model.get(target_img)
    source_faces = swapper.det_model.get(source_img)
    
    if len(target_faces) == 0 or len(source_faces) == 0:
        print("  警告: 未检测到人脸")
    else:
        # 绘制检测框
        detected_img = target_img.copy()
        for face in target_faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(detected_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)
            # 绘制关键点
            kps = face.kps.astype(int)
            for kp in kps:
                cv2.circle(detected_img, tuple(kp), 2, (0, 0, 255), -1)
        
        # 执行替换
        swapped_path = swapper.run(target_path, dst_face_paths=None, src_face_paths=source_path, output_dir=output_dir)
        swapped_img = cv2.imread(swapped_path)
        
        # 创建流程图
        process_images = [
            source_img,
            target_img,
            detected_img,
            swapped_img
        ]
        process_labels = [
            '1.源人脸',
            '2.目标图像',
            '3.人脸检测',
            '4.替换结果'
        ]
        
        create_comparison_grid(
            process_images,
            process_labels,
            f"{output_dir}/process_flow.png",
            cols=4
        )
    
    # ========================================
    # 实验4: 多人脸场景测试
    # ========================================
    multi_path = "docs/test/multi.png"
    if os.path.exists(multi_path):
        print("\n[实验4] 测试多人脸场景...")
        
        multi_img = cv2.imread(multi_path)
        swapper = FaceSwapper(face_det_model='buffalo_l', use_enhancer=True, use_sr=False)
        
        # 检测多人脸
        faces = swapper.det_model.get(multi_img)
        print(f"  检测到 {len(faces)} 个人脸")
        
        # 绘制检测结果
        detected_multi = multi_img.copy()
        for i, face in enumerate(faces):
            bbox = face.bbox.astype(int)
            cv2.rectangle(detected_multi, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)
            cv2.putText(detected_multi, f"Face {i+1}", (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 替换结果
        swapped_multi_path = swapper.run(multi_path, dst_face_paths=None, src_face_paths=source_path, output_dir=output_dir)
        swapped_multi = cv2.imread(swapped_multi_path)
        
        multi_images = [multi_img, detected_multi, swapped_multi]
        multi_labels = ['原始多人图像', f'检测结果({len(faces)}个人脸)', '批量替换结果']
        
        create_comparison_grid(
            multi_images,
            multi_labels,
            f"{output_dir}/multi_face_test.png",
            cols=3
        )
    
    # ========================================
    # 生成图像信息文件
    # ========================================
    info_file = f"{output_dir}/图像说明.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("论文实验效果图说明\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"源人脸: {source_path}\n")
        f.write(f"目标图像: {target_path}\n\n")
        f.write("生成的图像文件:\n")
        f.write("1. config_comparison.png - 展示不同配置的效果差异\n")
        f.write("2. ablation_comparison.png - 展示各模块的贡献\n")
        f.write("3. process_flow.png - 展示完整处理步骤\n")
        f.write("4. multi_face_test.png - 展示批量处理能力\n\n")
        f.write("使用方法:\n")
        f.write("在论文中使用Markdown语法引用图像:\n")
        f.write("![图像说明](output/paper_figures/图像文件名.png)\n")
    
    print(f"\n✓ 所有对比图生成完成！")
    print(f"✓ 输出目录: {output_dir}")
    print(f"✓ 图像说明: {info_file}")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
