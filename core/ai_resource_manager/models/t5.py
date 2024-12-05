import os

import ml2rt
import numpy as np

from configs.common import BASE_DIR
from core.ai_resource_manager import RedisAIModel
from core.utils import T5InferenceHelper


class T5RedisModel(RedisAIModel):
    def initiate(
        self,
        backend: str = "torch",
        device: str = "cpu",
        encoder_path: str = "",
        decoder_path: str = "",
    ):
        if not encoder_path:
            encoder_path = os.path.join(BASE_DIR, "assets/encoder.pt")
        if not decoder_path:
            decoder_path = os.path.join(BASE_DIR, "assets/decoder.pt")

        en_model = ml2rt.load_model(encoder_path)
        de_model = ml2rt.load_model(decoder_path)

        self.store_model(key="encoder", backend=backend, device=device, data=en_model)
        self.store_model(key='decoder', backend=backend, device=device, data=de_model)

    def process(self, indices: list, **kwargs):
        nparray = T5InferenceHelper.list2numpy(indices)

        # Set input tensor
        self.feed_model(key="sentence", tensor=nparray)
        self.feed_model(key="length", tensor=np.array([nparray.shape[0]]).astype(np.int64))

        # Execute model
        self.execute_model(key='encoder', inputs=['sentence', 'length'], outputs=['e_output', 'hidden'])

        # Decoding steps
        hidden = self.get_tensor_model("hidden")[:2]
        self.feed_model(key="hidden", tensor=hidden)

        inter_tensor = np.array(T5InferenceHelper.SOS_token, dtype=np.int64).reshape(1, 1)
        self.feed_model(key="d_input", tensor=inter_tensor)

        out: list = []
        for i in range(self.max_tokens):
            self.execute_model(
                key='decoder',
                inputs=['d_input', 'hidden', 'e_output'],
                outputs=['d_output', 'hidden']
            )
            d_output = self.get_tensor_model(key="d_output")

            ind = int(d_output.argmax())
            if ind == T5InferenceHelper.EOS_token:
                break
            inter_tensor = np.array(ind, dtype=np.int64).reshape(1, 1)
            self.feed_model(key="d_input", tensor=inter_tensor)

            if ind == T5InferenceHelper.PAD_token:
                continue
            out.append(ind)
        return T5InferenceHelper.indices2str(out)
