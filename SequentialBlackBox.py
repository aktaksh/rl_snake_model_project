from RLearningSnake import run
from GPyOpt.methods import BayesianOptimization
import datetime

class BayesianOptimizer():
    def __init__(self, params):
        self.params = params

# Gradient Boosting Machine
    def gbm_cl_bo(max_depth, max_features, learning_rate, n_estimators, subsample):
        params_gbm = {}
        params_gbm['max_depth'] = round(max_depth)
        params_gbm['max_features'] = max_features
        params_gbm['learning_rate'] = learning_rate
        params_gbm['n_estimators'] = round(n_estimators)
        params_gbm['subsample'] = subsample
        scores = cross_val_score(GradientBoostingClassifier(random_state=123, **params_gbm),
                                X_train, y_train, scoring=acc_score, cv=5).mean()
        score = scores.mean()
        return score

    def bayesian_optimization(n_iters, sample_loss, xp, yp):
        """

        Arguments:
        ----------
            n_iters: int.
            Number of iterations to run the algorithm for.
            sample_loss: function.
            Loss function that takes an array of parameters.
            xp: array-like, shape = [n_samples, n_params].
            Array of previously evaluated hyperparameters.
            yp: array-like, shape = [n_samples, 1].
            Array of values of `sample_loss` for the hyperparameters
            in `xp`.
        """

        # Define the GP
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=1e-4,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)
        for i in range(n_iters):
            # Update our belief of the loss function
            model.fit(xp, yp)

            # sample_next_hyperparameter is a method that computes the arg
            # max of the acquisition function
            next_sample = sample_next_hyperparameter(model, yp)

            # Evaluate the loss for the new hyperparameters
            next_loss = sample_loss(next_sample)

            # Update xp and yp


    def optimize_RL(self):
        def optimize(inputs):
            print("INPUT", inputs)
            inputs = inputs[0]

            # Variables to optimize
            self.params["learning_rate"] = inputs[0]
            lr_string = '{:.8f}'.format(self.params["learning_rate"])[2:]
            self.params["first_layer_size"] = int(inputs[1])
            self.params["second_layer_size"] = int(inputs[2])
            self.params["third_layer_size"] = int(inputs[3])
            self.params["epsilon_decay_linear"] = int(inputs[4])

            self.params['name_scenario'] = 'snake_lr{}_struct{}_{}_{}_eps{}'.format(lr_string,
                                                                               self.params['first_layer_size'],
                                                                               self.params['second_layer_size'],
                                                                               self.params['third_layer_size'],
                                                                               self.params['epsilon_decay_linear'])

            self.params['weights_path'] = 'weights/weights_' + self.params['name_scenario'] + '.h5'
            self.params['load_weights'] = False
            self.params['train'] = True
            print(self.params)
            score, mean, stdev = run(self.params)
            print('Total score: {}   Mean: {}   Std dev:   {}'.format(score, mean, stdev))
            with open(self.params['log_path'], 'a') as f: 
                f.write(str(self.params['name_scenario']) + '\n')
                f.write('Params: ' + str(self.params) + '\n')
            return score

        optim_params = [
            {"name": "learning_rate", "type": "continuous", "domain": (0.00005, 0.001)},
            {"name": "first_layer_size", "type": "discrete", "domain": (20,50,100,200)},
            {"name": "second_layer_size", "type": "discrete", "domain": (20,50,100,200)},
            {"name": "third_layer_size", "type": "discrete", "domain": (20,50,100,200)},
            {"name":'epsilon_decay_linear', "type": "discrete", "domain": (self.params['episodes']*0.2,
                                                                           self.params['episodes']*0.4,
                                                                           self.params['episodes']*0.6,
                                                                           self.params['episodes']*0.8,
                                                                           self.params['episodes']*1)}
        ]

        bayes_optimizer = BayesianOptimization(f=optimize,
                                               domain=optim_params,
                                               initial_design_numdata=6,
                                               acquisition_type="EI",
                                               exact_feval=True,
                                               maximize=True)

        bayes_optimizer.run_optimization(max_iter=20)
        print('Optimized learning rate: ', bayes_optimizer.x_opt[0])
        print('Optimized first layer: ', bayes_optimizer.x_opt[1])
        print('Optimized second layer: ', bayes_optimizer.x_opt[2])
        print('Optimized third layer: ', bayes_optimizer.x_opt[3])
        print('Optimized epsilon linear decay: ', bayes_optimizer.x_opt[4])
        return self.params


##################
#      Main      #
##################
if __name__ == '__main__':
    # Define optimizer
    SequentialBlackBox = BayesianOptimizer(params)
    SequentialBlackBox.optimize_RL()

# #Run Bayesian Optimization
# start = time.time()
# params_gbm ={
#     'max_depth':(3, 10),
#     'max_features':(0.8, 1),
#     'learning_rate':(0.01, 1),
#     'n_estimators':(80, 150),
#     'subsample': (0.8, 1)
# }
# gbm_bo = BayesianOptimization(gbm_cl_bo, params_gbm, random_state=111)
# gbm_bo.maximize(init_points=20, n_iter=4)
# print('It takes %s minutes' % ((time.time() - start)/60))