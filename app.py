import gradio as gr
# from PIL import Image
# from modelscope_studio import encode_image, decode_image, call_demo_service
import json
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2
import face_preprocess
import imageio

###########################################
# 调试参数
###########################################
is_local_debug = False
debug_det_result_dir = './debug_det_result/'
debug_example_dir = './new_pic_wangzai/'

def save_det_result(img_path, src_img, raw_result):
    from modelscope.utils.cv.image_utils import draw_face_detection_result
    if not os.path.exists(debug_det_result_dir):
        os.makedirs(debug_det_result_dir)
        
    debug_src_img_path = debug_det_result_dir + 'det_src_' + os.path.split(img_path)[1]
    # save src img
    cv2.imwrite(debug_src_img_path, src_img)
    # save det result img
    dst_img = draw_face_detection_result(debug_src_img_path, raw_result)
    cv2.imwrite(debug_det_result_dir + 'det_result_' + os.path.split(img_path)[1], dst_img)
    
###########################################
# 全局超参
###########################################
online_example_dir = './images/'
online_example_id = ['id1/', 'id2/']
is_compress_input_img = True
# 视频参数        
video_fps = 5        
video_size = [224, 256]
face_template_size = [192, 224]
# 输入图像参数
compress_large_edge = 1280
# 融合渐变次数
blend_num = 3
# 原图停留次数
stop_num = 2

def compress_input_img(src_img):
    dst_width = src_img.shape[1] if compress_large_edge > src_img.shape[1] else compress_large_edge
    dst_height = int(src_img.shape[0] * dst_width / src_img.shape[1])
    src_img = cv2.resize(src_img, (dst_width, dst_height), interpolation=cv2.INTER_AREA)
    return src_img

###########################################
# gradio demo app 推断入口
###########################################

# gradio app demo 算法运行函数
def inference(input_files, input_param_gif_fps, input_param_gif_width, input_param_gif_height):
    video_fps = int(input_param_gif_fps)
    video_size[0] = int(input_param_gif_width)
    video_size[1] = int(input_param_gif_height)
    video_last_img = None
    gif_img_lst = []
    if input_files is None:
        return None
    input_files = [x.name for x in input_files]
    print(input_files)
    
    # 过程中间变量
    valid_img_num = 0
    
    # 绕过输入
    if is_local_debug :
        input_files = os.listdir(os.path.dirname(__file__) +  debug_example_dir)
        input_files.sort()
        input_files = [os.path.dirname(__file__) +  debug_example_dir + x for x in input_files]
    
    for i in range(0, len(input_files)):
        try:
            print(input_files[i])
            src_img_path = input_files[i]
            src_img = cv2.imread(src_img_path)
            if src_img is None:
                continue

            # # compress input image
            if is_compress_input_img :
                src_img = compress_input_img(src_img)
    
            # get face detection and landmark result
            facial_landmark_confidence_func = pipeline(Tasks.face_2d_keypoints, 'damo/cv_manual_facial-landmark-confidence_flcm')
            raw_result = facial_landmark_confidence_func(src_img)
            if raw_result is None:
                continue
            
            if float(raw_result['scores'][0]) < (1 - 0.145) :
                print('landmark quality fail...')
                continue
            # import pdb;pdb.set_trace()
            
            box_ldmk_str = ' '.join(str(x) for x in raw_result['boxes'][0]) \
            + ' ' + ' '.join(str(raw_result['keypoints'][0][x*2]) for x in range(5)) \
            + ' ' + ' '.join(str(raw_result['keypoints'][0][x*2+1]) for x in range(5))

            
            # if you want to show the result, you can run
            if is_local_debug :
                save_det_result(src_img_path, src_img, raw_result)

            # align face
            warped_face = face_preprocess.align_face(src_img, box_ldmk_str, face_template_size, video_size)
            if warped_face is None:
                print('warp ' + src_img_path + ' error')
                continue

            # 获得渐变效果
            if valid_img_num != 0:  
                last_face = video_last_img
                cur_face = warped_face
                
                for j in range(0, blend_num):
                    alpha = (j+1) / (blend_num+1)
                    beta = 1 - alpha
                    # face add weight
                    last_face = cv2.addWeighted(cur_face, alpha, last_face, beta, 0.)
                    gif_img_lst.append(cv2.cvtColor(last_face, cv2.COLOR_BGR2RGB))   
            # 获得停留效果
            for j in range(0, stop_num):
                gif_img_lst.append(cv2.cvtColor(warped_face, cv2.COLOR_BGR2RGB))

            video_last_img = warped_face
            valid_img_num += 1
        except Exception:
            print('process ' + input_files[i] + ' ' + Exception)
            pass
    if len(gif_img_lst) == 0:
        return None
    imageio.mimsave('output.gif', gif_img_lst, fps=video_fps)
    return 'output.gif'

# gradio app 环境参数
css_style = "#fixed_size_img {height: 240px;} " \
            "#overview {margin: auto;max-width: 600px; max-height: 400px;}"
title = "时光相册"
description = "<center> <p> 随意上传您心仪的照片合集（宝宝成长相册、同学毕业相册），通过人脸五官对齐，一键生成样貌变化效果图，点点鼠标即可下载分享给亲朋好友啦，立刻玩起来吧!!! </p> </center>"

###########################################
# gradio demo app
###########################################
with gr.Blocks(title=title, css=css_style) as demo:
    gr.HTML('''
      <div style="text-align: center; max-width: 720px; margin: 0 auto;">
                  <div
                    style="
                      display: inline-flex;
                      align-items: center;
                      gap: 0.8rem;
                      font-size: 1.75rem;
                    "
                  >
                    <h1 style="font-family:  PingFangSC; font-weight: 500; line-height: 1.5em; font-size: 32px; margin-bottom: 7px;">
                      时光相册
                    </h1>
                  </div>
                  <img id="overview" alt="overview" src="https://mogface.oss-cn-zhangjiakou.aliyuncs.com/modelscope/face_album/template.gif" />
                  
                </div>
      ''')


    gr.Markdown(description)
    with gr.Row():
        input_param_gif_fps = gr.Slider(1, 32, step =1, value =5, label="视频帧率（数值越大播放越快，非常有趣，试试看！！！）" )
        input_param_gif_width = gr.Slider(224, 1280, step =8, value =224, label="视频宽度（数值越大横向背景越多）" )
        input_param_gif_height = gr.Slider(256, 1280, step= 8, value=256,label="视频高度（数值越大纵向背景越多）")
    with gr.Row():
        input_files = gr.Files(elem_id="fixed_size_img", label="图片合集（上传同一人在不同时期的照片，暂不支持补、删图，修改合集建议刷新或清除后重传）")
        img_output = gr.Image(type="filepath", elem_id="fixed_size_img", label="视频结果")
    with gr.Row():
        btn_submit = gr.Button(value="一键生成", elem_id="blue_btn")

    # append all examples
    examples = []
    for i in range(0, len(online_example_id)):   
        example_id_dir = os.path.dirname(__file__) + online_example_dir + online_example_id[i]
        example_id_names = os.listdir(example_id_dir)
        example_id_names.sort()
        example_id_names = [example_id_dir + x for x in example_id_names]
        examples.append([example_id_names])
    
    print(examples)
    examples = gr.Examples(examples=examples, inputs=input_files, outputs=img_output, label="点击如下示例试玩", run_on_click=False)
    btn_submit.click(inference, inputs=[input_files, input_param_gif_fps, input_param_gif_width, input_param_gif_height], outputs=img_output)
    # btn_clear清除画布


if __name__ == "__main__":
    demo.launch(share=False)
