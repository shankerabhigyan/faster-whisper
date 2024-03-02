import logging

from .base import ASRBase


class FasterWhisperASR(ASRBase):
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version.

    Requires imports, if used:
        import faster_whisper
    """

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None):
        super().__init__(lan, modelsize, cache_dir, model_dir)
        
    sep = ""

    def get_model_size_or_path(self, modelsize=None, model_dir=None):
        if model_dir is not None:
            return model_dir
        elif modelsize is not None:
            return modelsize
        else:
            return None
    
    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel
        
        if model_dir is not None:
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
            
        if model_size_or_path := self.get_model_size_or_path(modelsize=modelsize, model_dir=model_dir):
            logging.info(f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used.")
            return WhisperModel(model_size_or_path, device="cuda", compute_type="float16", download_root=cache_dir)    
        else:
            raise ValueError("modelsize or model_dir parameter must be set")
        

    def transcribe(self, audio, init_prompt=""):
        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(audio, language="tl", initial_prompt=init_prompt, beam_size=5, word_timestamps=True, condition_on_previous_text=True, **self.transcribe_kargs)
        return list(segments)

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"
