from matplotlib import pyplot as plt


def show_img(img):
    #Show the image with matplotlib

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.close()

