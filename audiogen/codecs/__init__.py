from audiogen.codecs.base import CodecBase, CodecOutput
from audiogen.codecs.snac import SNACCodec
from audiogen.codecs.moss import MOSSCodec


def load_codec(name: str, **kwargs) -> CodecBase:
    if name == "snac":
        return SNACCodec.load(**kwargs)
    elif name == "moss":
        return MOSSCodec.load(**kwargs)
    else:
        raise ValueError(f"Unknown codec: {name!r}. Choose 'snac' or 'moss'.")


__all__ = ["CodecBase", "CodecOutput", "SNACCodec", "MOSSCodec", "load_codec"]
