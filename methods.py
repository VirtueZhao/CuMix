

class CuMix:

    def __init__(self, seen_classes, unseen_classes, attributes, configs, zsl_only=False, dg_only=False,
                 device='cuda', word_size=1, rank=0):
        self.end_to_end = True
        self.domain_mix = True

        print("Hi")
