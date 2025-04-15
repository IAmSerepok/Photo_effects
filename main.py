from photo_filters import PhotoFilters as App


if __name__ == "__main__":
    app = App(input_path="input/test.png")
    app.convolution('box_blur')
    # app.save_image('out.png')
    app.run()
