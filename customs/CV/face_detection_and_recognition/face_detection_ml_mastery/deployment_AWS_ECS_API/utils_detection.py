from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import base64


##############################################################################
def draw_image_with_boxes(filename, result_list):
    """
    Function to draw bounding box and key points around faces in an image
    :param filename: File path (image)
    :param result_list: Output from MTCNN class
    :return: Saves result.png with bounding box and key points
    """
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    # show the plot
    pyplot.savefig('result.png')


def draw_faces(filename, result_list):
    """
    Function to draw faces from an input image
    :param filename: File path (image)
    :param result_list: Output from MTCNN class
    :return: Saves result.png with faces detected
    """
    # load the image
    data = pyplot.imread(filename)
    # plot each face as a subplot
    for i in range(len(result_list)):
        # get coordinates
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        # define subplot
        pyplot.subplot(1, len(result_list), i + 1)
        pyplot.axis('off')
        # plot face
        pyplot.imshow(data[y1:y2, x1:x2])
    # show the plot
    pyplot.savefig('result.png')


def write_to_file(save_path, data):
    """
    This function writes input image to temporary file for further processing
    :param save_path: Output path
    :param data: Input image
    :return: None
    """
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(data))


def allowed_file(filename, ALLOWED_EXTENSIONS):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
