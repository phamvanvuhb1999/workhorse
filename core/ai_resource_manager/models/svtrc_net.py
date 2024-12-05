import os

import ml2rt
import numpy as np

from configs.common import BASE_DIR
from core.ai_resource_manager import RedisAIModel
from core.ai_resource_manager.models.utils.postprocess import CTCLabelDecode
from core.ai_resource_manager.models.utils.preprocess import resize_with_padding


class SVTRCLNetRecognizerRedisModel(RedisAIModel):
    model_key = "svtrcl_net"

    def __init__(
        self,
        rec_batch_num: int = 12,
        character_dict_path: str = None,
        use_space_char: bool = True,
        rec_image_shape: list | None = None,
        drop_score: float = 0.5,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if not rec_image_shape:
            rec_image_shape = [3, 48, 320]
        if not character_dict_path:
            character_dict_path = os.path.join(BASE_DIR, "assets/ocr_keys.txt")

        input_tensor_name = kwargs.get("input_tensor_name", "x")
        output_tensor_name = kwargs.get("output_tensor_name", "softmax_11.tmp_0")

        self.input_tensor_name = input_tensor_name
        self.output_tensor_name = output_tensor_name
        self.rec_batch_num = rec_batch_num
        self.rec_image_shape = rec_image_shape
        self.drop_score = drop_score
        self.decoder = CTCLabelDecode(character_dict_path=character_dict_path, use_space_char=use_space_char)


    def initiate(
        self,
        backend: str = "onnx",
        device: str = "cpu",
        model_path: str = "",
    ):
        if not model_path:
            model_path = os.path.join(BASE_DIR, "assets/text_recognition/PP_OCRv4_rec.onnx")

        en_model = ml2rt.load_model(model_path)

        self.store_model(key=self.model_key, backend=backend, device=device, data=en_model)

    def process(self, images: np.ndarray, dt_boxes: list, **kwargs):
        batch_num = self.rec_batch_num
        _, img_h, img_w = self.rec_image_shape[:3]

        img_num = len(images)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in images:
            width_list.append(img.shape[1] / float(img.shape[0]))

        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [["", 0.0]] * img_num

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = img_w / img_h

            for ino in range(beg_img_no, end_img_no):
                h, w = images[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                norm_img = resize_with_padding(
                    image=images[indices[ino]],
                    # max_wh_ratio=max_wh_ratio,
                )
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch)
            # norm_img_batch = np.resize(norm_img_batch, [*norm_img_batch.shape[:-1], 320])

            print(f"norm_img_batch shape {norm_img_batch.shape}")
            self.feed_model(key=self.input_tensor_name, tensor=norm_img_batch)
            self.execute_model(key=self.model_key, inputs=[self.input_tensor_name,], outputs=[self.output_tensor_name,])

            outputs = self.get_tensor_model(key=self.output_tensor_name)
            rec_result = self.decoder(outputs)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        # Release model lock
        # self.release_model_lock(**kwargs)

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            _text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        return {
            "bboxes": filter_boxes,
            "texts": filter_rec_res,
            "step": self.class_prefix(),
            "is_finished": True,
        }

if __name__ == "__main__":
    from redis import Redis
    import cv2 as cv
    from datetime import datetime

    # svt_rec = SVTRCLNetRecognizerRedisModel.get_instance(prefix="svtrclnetrecognizerredismodel:00e6e07b-f05b-45a5-8093-f62b745c7cae:svtrcl_net")
    # redis_cli = Redis()
    # if not redis_cli.keys("svtrclnetrecognizerredisdodel:*"):
    #     svt_rec.initiate()
    # imgg = cv.imread(r"/home/vupham/Downloads/WhatsApp Image 2024-12-03 at 15.07.55.jpeg", cv.IMREAD_GRAYSCALE)
    # imgg = cv.cvtColor(imgg, cv.COLOR_GRAY2BGR)
    # start_time = datetime.now()
    # print(f"start time {start_time}")
    # result = svt_rec.process(np.array([imgg]))
    # print(f"end time: {datetime.now()}, executed time {datetime.now() - start_time}")
    # print(result)

    svt_rec = SVTRCLNetRecognizerRedisModel.get_instance()
    svt_rec.initiate()