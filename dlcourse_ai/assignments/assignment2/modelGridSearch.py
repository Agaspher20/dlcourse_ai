import numpy as np

def initialize_model(model_factory, best_model_keys):
    model = model_factory()
    for best_key in best_model_keys:
        setattr(model, best_key, best_model_keys[best_key])

    return model

def search_model(
    model_factory,
    trainer_factory,
    model_grid,
    trainer_grid
):
    model = model_factory()
    trainer = trainer_factory(model)

    initial_values = trainer.fit()

    best_model_keys = {}
    best_loss_history, best_train_history, best_val_history = initial_values
    best_train = np.max(best_train_history)
    best_model = model

    for key in model_grid:
        best_model_keys[key] = getattr(best_model, key)
        
        for value in model_grid[key]:
            model = initialize_model(model_factory, best_model_keys)
            setattr(model, key, value)

            trainer = trainer_factory(model)
        
            # You should expect loss to go down and train and val accuracy go up for every epoch
            loss_history, train_history, val_history = trainer.fit()
            max_train = np.max(train_history)
        
            if max_train > best_train:
                best_train = max_train
                best_model_keys[key] = value
                best_loss_history = loss_history,
                best_train_history = train_history,
                best_val_history = val_history
                best_model = model
        print("\nBest {} is {} with accuracy {}\n".format(key, best_model_keys[key], best_train))

    best_trainer_keys = {}
    best_trainer = trainer_factory(model_factory())

    for key in trainer_grid:
        best_trainer_keys[key] = getattr(best_trainer, key)

        for value in trainer_grid[key]:
            model = initialize_model(model_factory, best_model_keys)

            trainer = initialize_model(lambda: trainer_factory(model), best_trainer_keys)
            setattr(trainer, key, value)
        
            # You should expect loss to go down and train and val accuracy go up for every epoch
            loss_history, train_history, val_history = trainer.fit()
            max_train = np.max(train_history)
        
            if max_train > best_train:
                best_train = max_train
                best_trainer_keys[key] = value
                best_loss_history = loss_history,
                best_train_history = train_history,
                best_val_history = val_history
        print("\nBest {} is {} with accuracy {}\n".format(key, best_trainer_keys[key], best_train))

    return (
        best_model,
        best_model_keys,
        best_trainer_keys,
        best_loss_history,
        best_train_history,
        best_val_history
    )
