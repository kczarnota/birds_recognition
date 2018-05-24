import argparse

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument("-model", help="Path to model")
    parser.add_argument("-test", help="Path to test data")
    args = parser.parse_args()

    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(
        args.test,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical')
    model = load_model(args.model)
    scores = model.evaluate_generator(test_generator, 5)
    print(scores)
