from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks

ulfd_face_detection = pipeline(Tasks.face_detection, 'damo/cv_manual_face-detection_ulfd')
# img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ulfd_face_detection.jpg'
img_path = './images/face/people.jpeg'
# img_path = './images/face/sheep.jpg'
result = ulfd_face_detection(img_path)
print(f'face detection output: {result}.')