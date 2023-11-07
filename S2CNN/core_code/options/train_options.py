from options.base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.is_Train = True
        self.parser.add_argument('--batchsize', type=int, default=16,
                                help='input batch size')
        self.parser.add_argument('--lr', type=float, default=0.0002,
                                help='learning rate')
        self.parser.add_argument('--epsilon', type=float, default=1e-9,
                                help='epsilon')
        self.parser.add_argument('--lambda_adv', type=float, default=1,
                                help='weight of G/D adv loss')
        self.parser.add_argument('--lambda_expr', type=float, default=10,
                                help='weight of G expr loss')
        self.parser.add_argument('--lambda_id', type=float, default=200,
                                help='weight of G id loss')
        self.parser.add_argument('--lambda_p', type=float, default=0.01,
                                help='weight of G p loss')
        self.parser.add_argument('--lambda_gp', type=float, default=10,
                                help='weight of D gp loss')
        self.parser.add_argument('--lambda_reg', type=float, default=0.00001,
                                help='weight of G p loss')
        self.parser.add_argument('--start_epoch', type=int, default=1,
                                help='strat num of count epoch')
        self.parser.add_argument('--end_epoch', type=int, default=11,
                                help='nums of eopch')
        self.parser.add_argument('--beta1', type=float, default=0.5, 
                                help='Adam parameter 1')
        self.parser.add_argument('--beta2', type=float, default=0.999,
                                help='Adam parameter 2')
        self.parser.add_argument('--save_freq', type=int, default=2,
                                help='frequency of saving model')
        self.parser.add_argument('--train_freq', type=int, default=1,
                                help='frequency of optimization')
        self.parser.add_argument('--reload_epoch', type=int, default=0,
                                help='reload of eopch')