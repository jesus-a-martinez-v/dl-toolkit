from keras.applications import VGG19

from nn.conv.neuralstyle import NeuralStyle

SETTINGS = {
    'input_path': 'inputs/pug.jpg',
    'style_path': 'inputs/starry_night.jpg',
    'output_path': 'output',

    # CNN to be used for style transfer, along with the set of content
    # layer and style layers, respectively
    'net': VGG19,
    'content_layer': 'block4_conv2',
    'style_layers': ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],

    # Store the content, style and total variation weights, respectively
    'content_weight': 1.0,
    'style_weight': 100.0,
    'tv_weight': 10.0,

    # Number of iterations
    'iterations': 50
}

neural_style = NeuralStyle(SETTINGS)
neural_style.transfer()