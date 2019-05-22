import argparse

from keras.applications import VGG16

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--include-top', type=int, default=1, help='Whether or not to include top of CNN.')
arguments = vars(argument_parser.parse_args())

print('[INFO] Loading network...')
model = VGG16(weights='imagenet', include_top=arguments['include_top'] > 0)
print('[INFO] Showing layers...')

for i, layer in enumerate(model.layers):
    print(f'[INFO] {i}\t{layer.__class__.__name__}')
