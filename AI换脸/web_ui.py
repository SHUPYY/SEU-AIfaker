import argparse
import gradio as gr
from dofaker import FaceSwapper


def parse_args():
    parser = argparse.ArgumentParser(description="running face swap")
    parser.add_argument(
        "--inbrowser",
        help="whether to automatically launch the interface in a new tab on the default browser.",
        dest="inbrowser",
        default=True,
    )
    parser.add_argument(
        "--server_port",
        help=(
            "will start gradio app on this port (if available)."
            "Can be set by environment variable GRADIO_SERVER_PORT."
            "If None, will search for an available port starting at 7860."
        ),
        dest="server_port",
        type=int,
        default=None,
    )
    return parser.parse_args()


def swap_face(
    input_path, dst_path, src_path, use_enhancer, use_sr, scale, face_sim_thre
):
    faker = FaceSwapper(
        use_enhancer=use_enhancer,
        use_sr=use_sr,
        scale=scale,
        face_sim_thre=face_sim_thre,
    )
    output_path = faker.run(input_path, dst_path, src_path)
    return output_path


def main():
    args = parse_args()

    with gr.Blocks(title="DoFaker") as web_ui:
        gr.Markdown(
            "<div style='font-size:22px; margin-bottom:0px;'>王小康-255591</div>"
        )
        with gr.Tab("图片"):
            gr.Markdown("<div style='font-size:15px; margin:0px;'>王小康-255591</div>")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("源图片")
                    image_input = gr.Image(type="filepath")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("目标人脸（被换）")
                            dst_face_image = gr.Image(type="filepath")
                        with gr.Column():
                            gr.Markdown("用于替换的人脸")
                            src_face_image = gr.Image(type="filepath")

                with gr.Column():
                    output_image = gr.Image(type="filepath")
                    use_enhancer = gr.Checkbox(label="面部增强", info="是否使用面部增强模型")
                    with gr.Row():
                        use_sr = gr.Checkbox(label="超分辨率", info="是否使用图像分辨率模型")
                        scale = gr.Number(value=1, label="图像超分比例")
                    with gr.Row():
                        face_sim_thre = gr.Number(
                            value=0.6,
                            label="人脸相似度阈值",
                            minimum=0.0,
                            maximum=1.0,
                            visible=False,
                        )
                    convert_button = gr.Button("Swap")
                    convert_button.click(
                        fn=swap_face,
                        inputs=[
                            image_input,
                            dst_face_image,
                            src_face_image,
                            use_enhancer,
                            use_sr,
                            scale,
                            face_sim_thre,
                        ],
                        outputs=[output_image],
                        api_name="image swap",
                    )

        with gr.Tab("视频"):
            gr.Markdown("王小康-255591")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("源视频")
                    video_input = gr.Video()
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("用来替换的人脸")
                            src_face_image = gr.Image(type="filepath")
                with gr.Column():
                    output_video = gr.Video()
                    use_enhancer = gr.Checkbox(label="面部增强", info="是否使用面部增强模型")
                    with gr.Row():
                        use_sr = gr.Checkbox(
                            label="超分辨率",
                            info=" 是否使用图像分辨率模型",
                        )
                        scale = gr.Number(value=1, label="图像超分比例")
                    with gr.Row():
                        face_sim_thre = gr.Number(
                            value=0.6,
                            label="人脸相似度阈值",
                            minimum=0.0,
                            maximum=1.0,
                            visible=False,
                        )
                    convert_button = gr.Button("Swap")
                    convert_button.click(
                        fn=swap_face,
                        inputs=[
                            video_input,
                            dst_face_image,
                            src_face_image,
                            use_enhancer,
                            use_sr,
                            scale,
                            face_sim_thre,
                        ],
                        outputs=[output_video],
                        api_name="video swap",
                    )

    web_ui.launch(inbrowser=args.inbrowser, server_port=args.server_port)


if __name__ == "__main__":
    main()
