import sys
from options_py.base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.is_Train=False
        self.parser.add_argument('--batchsize', type=int, default=1,
                                help='input batch size')