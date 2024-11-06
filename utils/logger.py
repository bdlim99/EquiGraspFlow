class Logger:
    def __init__(self, writer):
        self.writer = writer

    def log(self, results, iter):
        for key, val in results.items():
            if 'scalar' in key:
                self.writer.add_scalar(key.replace('scalar/', ''), val, iter)

            elif 'image' in key and 'images' not in key:
                self.writer.add_image(key.replace('image/', ''), val, iter)

            elif 'images' in key:
                self.writer.add_images(key.replace('images/', ''), val, iter)

            elif 'mesh' in key:
                self.writer.add_mesh(key.replace('mesh/', ''), vertices=val['vertices'], colors=val['colors'], faces=val['faces'], global_step=iter)
