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
        image_height, image_width, c = img.shape
        if c == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # get bounding box and facial landmarks
        boxes = self.detector(img)
        landmarks = []
        for box in boxes:
            shape = self.shape_predictor(img, box)
            index = 1
            center_point = [shape.parts()[5].x + shape.parts()[48].x,  shape.parts()[5].y + shape.parts()[48].y ]
         
            cl = img[int(center_point[1]/2), int(center_point[0]/2)] # face random color
            maxy = shape.parts()[24].y # eyebrow
            maxy_2 = shape.parts()[19].y # eyebrow

            # 얼굴 가장 좌, 우 지점
            x0 = shape.parts()[0].x
            y0 = shape.parts()[0].y
            x16 = shape.parts()[16].x
            y16 = shape.parts()[16].y

            # box for "good" face skin area.
            mx, my, MX, MY = shape.parts()[47].x + 20, shape.parts()[47].y + 20, shape.parts()[14].x - 10, shape.parts()[14].y - 10
            pts1 = np.float32([[mx, my],[mx, MY],[MX, my],[MX, MY]])
            pts2 = np.float32([[0, 0],[0, 512],[512, 0],[512, 512]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(img, M, (512,512))
            print("mx : {}, my : {}, mx : {}, mY : {}", mx, my, MX, MY)
            print("my : ", my)
            print("MX : ", MX)
            print("MY : ", MY)

            for part in shape.parts():
                landmarks.append(eos.core.Landmark(str(index),[float(part.x),float(part.y)]))
                index +=1

            break # 첫 번째 얼굴만 적용

        (mesh, pose, shape_coeffs, blendshape_coeffs) = eos.fitting.fit_shape_and_pose(self.morphablemodel_with_expressions,
            landmarks, self.landmark_mapper, image_width, image_height, self.edge_topology, self.contour_landmarks, self.model_contour)

        # read segmetation mask
        seg = cv2.imread(mask) # 0 : background , 127 : hair, 254 : face // grayscale image
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
        
        # need to up-sample mask so that mask is same size with input image.
        seg_alias = cv2.resize(seg, (img.shape[0], img.shape[1]))
        seg = cv2.resize(seg, (img.shape[0], img.shape[1]),interpolation=cv2.INTER_NEAREST) # no anti-alising
        face_boarder = seg != seg_alias 

        # mask
        background = seg == 0
        hair = seg == 127
        face = seg == 254

        # find donminant face color..
        #pixels = np.float32(img[face].reshape(-1, 3))
        
        #pixels = img[face].astype('float32')
        #n_colors = 5
        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        #flags = cv2.KMEANS_RANDOM_CENTERS

        #_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        #_, counts = np.unique(labels, return_counts=True)
        
        # option 1
        #dominant = palette[np.argmax(counts)] # dominant color.
        
        # option 2
        #dominant = np.average(palette, axis=0 ,weights=counts)
        
        # option 3 : any face color...
        
        hair_dominant = np.average(img[hair], axis=0).astype('uint8')
        
        # option 4 : face color
        # dominant = cl
        # key color
        key_color = [7, 28, 98]
        hair_color = [28, 7, 98]
        boarder_color = [98, 7, 28]


        img_copy = np.copy(img)
        img_copy[face] = key_color

        img[hair] = key_color
        img[background] = key_color
        img[face_boarder] = boarder_color


        #앞머리
        img[ : max(0, maxy-150) , :] = key_color
        img[ : max(0, maxy_2 - 150), :] = key_color
        

        isomap = eos.render.extract_texture(mesh, pose, img)
        isomap_copy = eos.render.extract_texture(mesh, pose, img_copy)
        # cv2.imwrite(o[:-4] + ".isomap.png", isomap)
        isomap = cv2.transpose(isomap)
        isomap_copy = cv2.transpose(isomap_copy)

        empty = np.all( isomap == [0, 0, 0, 0], axis=-1)
        key_color = [7, 28, 98, 255]
        key_hair_color = [28, 7, 98, 255]
        key_boarder_color = [98, 7, 28, 255]

        kc = np.all( isomap == key_color, axis = -1)
        #kc_hair = np.all( isomap == key_hair_color, axis = -1)
        kc_boarder = np.all( isomap == key_boarder_color, axis = -1)
        kc_boarder_copy = np.all( isomap_copy == key_color, axis = -1)

        use_dst = kc 
        mask = kc_boarder
        mask_face = kc_boarder_copy

        mask_gray = np.zeros_like(mask).astype('uint8')
        mask_gray[mask] = 255
        kernel = np.ones((5,5), np.uint8)
        mask_gray = cv2.dilate(mask_gray, kernel, iterations=7)
        mask_gray[empty] = 255
        mask_gray[mask_face] = 0
        kc_boarder = np.all( isomap == key_boarder_color, axis = -1)
        mask_gray[kc_boarder] = 255
        #isomap[empty] = (dominant[0], dominant[1], dominant[2], 255)
        
        eos.core.write_textured_obj(mesh, o[:-4] + ".obj") # ex) <django_project_root_path>/media/recon_result/user_image.obj

        isomap = cv2.cvtColor(isomap, cv2.COLOR_RGBA2RGB)
        isomap[use_dst] = dst[use_dst] 

        cv2.imwrite("temp.png", cv2.cvtColor(isomap, cv2.COLOR_RGB2RGBA))
        cv2.imwrite("temp_mask.png", mask_gray)

        isomap = cv2.inpaint(isomap, mask_gray, 21, cv2.INPAINT_TELEA) # kernel size (third parameter) could be lower to reduce time delay.

        cv2.imwrite(o[:-4] + "_isomap.png", cv2.cvtColor(isomap, cv2.COLOR_RGB2RGBA)) # ex) <django_project_root_path>/media/recon_result/user_image.isomap.png
        # print(face_color.shape)
        # print(np.average(face_color[:][:][0], axis=0))
        # print(np.average(face_color[:][:][1], axis=0))
        # print(np.average(face_color[:][:][2], axis=0))
        # print(hair_color)
        cv2.imwrite(o[:-4] + "_scalp.png",  dst) # (b,g,r) face color image
        return (hair_dominant[2], hair_dominant[1], hair_dominant[0]) # (r,g,b) of hair color