
'''
This file contains the configs used for Model creation and training. You need to give your best hyperparameters and the configs you used to get the best results for 
every environment and experiment.  These configs will be automatically loaded and used to create and train your model in our servers.
'''
#You can add extra keys or modify to the values of the existing keys in bottom level of the dictionary.
#DO NOT CHANGE THE STRUCTURE OF THE DICTIONARY. 

configs = {
    
    'Hopper-v4': {
            #You can add or change the keys here
              "hyperparameters": {
                'hidden_size': 100,
                'n_layers': 6,
                'batch_size': 8000, 
                "num_traj" : 50,
                "max_len" : 5000,
                "lr" : 0.0001,
                "env_name" : "Hopper-v4"
            },
            "num_iteration": 1000,
    },
    
    
    'Ant-v4': {
            #You can add or change the keys here
              "hyperparameters": {
                #   "ntraj" : 50,
                # "maxtraj" : 2000,
                # "save" : False,
                'hidden_size': 100,
                'n_layers': 6,
                'batch_size': 1000, 
                "num_traj" : 200,
                "max_len" : 5000,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                "lr" : 0.001,
                "env_name" : "Ant-v4"
            },
            "num_iteration": 1000,
    }

}