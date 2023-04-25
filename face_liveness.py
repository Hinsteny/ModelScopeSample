from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

face_liveness_ir = pipeline(Tasks.face_liveness, 'damo/cv_manual_face-liveness_flir')
# img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_liveness_ir.jpg'
img_path = './images/people/red_girl.jpeg'
result = face_liveness_ir(img_path)
print(f'face liveness output: {result}.')