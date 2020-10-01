import numpy as np
import dlib
import eos
import cv2
import sys
import os

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
            print(1)
        else:
            cls._instances[cls].__init__(*args, **kwargs)
            print(2)
            
        return cls._instances[cls]

class FaceRecon(metaclass=Singleton):
    cnt = 0
    def __init__(self, root_path):
        if(FaceRecon.cnt == 0):
            FaceRecon.cnt += 1
            print(3)
            
            # load detector,shape predictor
            detector = dlib.get_frontal_face_detector()
            shape_predictor = dlib.shape_predictor(os.path.join(root_path, 'res/68.dat'))

            # load eos model
            model = eos.morphablemodel.load_model(os.path.join(root_path, "res/sfm_shape_3448.bin"))
            blendshapes = eos.morphablemodel.load_blendshapes(os.path.join(root_path, "res/expression_blendshapes_3448.bin"))

            # Create a MorphableModel with expressions from the loaded neutral model and blendshapes:
            morphablemodel_with_expressions = eos.morphablemodel.MorphableModel(model.get_shape_model(), blendshapes,
                                                                                color_model=eos.morphablemodel.PcaModel(),
                                                                                vertex_definitions=None,
                                                                                texture_coordinates=model.get_texture_coordinates())
            landmark_mapper = eos.core.LandmarkMapper(os.path.join(root_path, 'res/ibug_to_sfm.txt'))
            edge_topology = eos.morphablemodel.load_edge_topology(os.path.join(root_path, 'res/sfm_3448_edge_topology.json'))
            contour_landmarks = eos.fitting.ContourLandmarks.load(os.path.join(root_path, 'res/ibug_to_sfm.txt'))
            model_contour = eos.fitting.ModelContour.load(os.path.join(root_path, 'res/sfm_model_contours.json'))

            self.detector = detector
            self.shape_predictor = shape_predictor
            self.morphablemodel_with_expressions = morphablemodel_with_expressions
            self.landmark_mapper = landmark_mapper
            self.edge_topology = edge_topology
            self.contour_landmarks = contour_landmarks
            self.model_contour = model_contour

    def generate(self, i, o, mask) : # input name, output name, root path of recon code
        # load image
        img = cv2.imread(i) # i : cropped square image. # ex) i : <django_project_root_path>/media/img/user_image.jpg
        image_height, image_width, _ = img.shape

        # get bounding box and facial landmarks
        boxes = self.detector(img)
        landmarks = []
        for box in boxes:
            shape = self.shape_predictor(img, box)
            index = 1
            for part in shape.parts():
                landmarks.append(eos.core.Landmark(str(index),[float(part.x),float(part.y)]))
                index +=1

        (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(self.morphablemodel_with_expressions,
            landmarks, self.landmark_mapper, image_width, image_height, self.edge_topology, self.contour_landmarks, self.model_contour)

        # read segmetation mask
        seg = cv2.imread(mask) # 0 : background , 127 : hair, 254 : face // grayscale image
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
        
        # need to up-sample mask so that mask is same size with input image.
        seg = cv2.resize(seg, (img.shape[0], img.shape[1]))

        # mask
        background = seg == 0
        hair = seg == 127
        face = seg == 254

        # find donminant face color..
        #pixels = np.float32(img[face].reshape(-1, 3))
        
        pixels = img[face].astype('float32')
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        
        # option 1
        dominant = palette[np.argmax(counts)] # dominant color.
        
        # option 2
        dominant = np.average(palette, axis=0 ,weights=counts)
        
        img[hair] = dominant
        img[background] = dominant
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        
        isomap = eos.render.extract_texture(mesh, pose, img)
        isomap = cv2.transpose(isomap)
        eos.core.write_textured_obj(mesh, o[:-4] + ".obj") # ex) <django_project_root_path>/media/recon_result/user_image.obj
        
        cv2.imwrite(o[:-4] + ".isomap.png", isomap) # ex) <django_project_root_path>/media/recon_result/user_image.isomap.png
