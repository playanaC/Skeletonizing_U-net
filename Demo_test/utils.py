import os
import cv2
import glob
import math
import numpy as np
from PIL import Image
from pathlib import Path
from xml.dom.minidom import parse
from math import pow, sqrt, floor
from torchvision import transforms
from skimage.measure import label, regionprops


def my_memcpy(dest, src, size, pd, ps):
    for i in range(0,size,1):
        dest[pd] = src[ps]
        ps += 1
        pd += 1
    return dest


def my_memset(des,val,nb,p):
    while nb > 0:
        des[p] = val
        p += 1
        nb -= 1
    return des


def reconstruir_circul(digital1, ample,alt):
    r = alt/2
    y0 = alt/2
    k2 = 0
    x0 = ample/2
    r2 = pow(r,2)

    NUM = y0 - r
    if NUM < 0:
        NUM = 0
    i1 = int(NUM * ample)
    digital2 = [0] * int(pow(1944,2))

    for j in range (0,alt,1):
        xs = int(floor(x0 - abs(sqrt(r2 - pow(r - j, 2)))))
        cx = int(floor(abs(sqrt(r2 - pow(r - j, 2)))))
        NUM = j * ample
        N1 = NUM + ample
        XN = NUM + xs
        Xc = XN + 2 * cx
        digital2 = my_memset(digital2, 100, XN - i1, i1)
        i1 = XN -1
        digital2 = my_memcpy(digital2,digital1,Xc - XN, i1, k2)
        i1 = i1 + Xc - XN
        k2 = k2 + Xc - XN
        digital2 = my_memset(digital2, 100, N1 - i1,i1)
        i1 = N1 - 1
    return np.asarray(digital2)


def read_image(path_image):
    extension = os.path.splitext(path_image)[1]
    if path_image.endswith('.cmpr'):
        with open(path_image, mode='rb') as file:
            fileContent = file.read()

        np_im = reconstruir_circul(fileContent, 1944, 1944)
        np_im = np_im.reshape(1944, 1944)
        new_im = Image.fromarray(np_im.astype('uint8'), 'L')
    else:
        new_im = Image.open(path_image).convert('L')
    return new_im


def build_skel(H, W, XY_skels):
    mask = np.zeros((H, W))
    for XY_skel in XY_skels:
        for XY in XY_skel:
            mask[XY[0], XY[1]] = 1
    return mask


def rz_norm_img(img, vm):
    img = (img > 0) * vm
    img = img.astype(np.uint8)
    return img


def get_concat_h(im1, im2):
    if im2.mode == 'RGB':
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
    else:
        dst = Image.new('L', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    if im2.mode == 'RGB':
        dst = Image.new('RGB', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
    else:
        dst = Image.new('L', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
    return dst


def xml_annotator(path_folder):
    annotator_files = []
    files = glob.glob(path_folder + '*.xml')

    for file in files:
        if "3_A_DT" in file:
            annotator_files.append(file)
    return annotator_files


def xml_points(xmls_annotator):

    docA = parse(xmls_annotator)
    items_XY = docA.getElementsByTagName('pose')

    skel = []
    for items in items_XY:
        XY_str = items.firstChild.nodeValue
        XY_str = XY_str.replace('[', '')
        XY_str = XY_str.replace(']', '')
        XY_list_str = XY_str.split(";")

        X = []
        Y = []
        for XY_p in XY_list_str:
            XY_list = XY_p.split()
            X.append(int(XY_list[0]))
            Y.append(int(XY_list[1]))

        points = np.transpose(np.vstack([Y, X]))
        skel.append(points)
    return skel


def img_cmpr_skl(data_files, xmls, class_img, ptsN):
    img = read_image(data_files)
    img = np.asarray(img)

    ext = data_files.split('.')[-1]
    split_data = data_files.split('.' + ext)[0]
    nn = int(split_data[len(split_data) - 2:])
    # path = Path(data_files)
    # father_folder = str(path.parent.absolute())

    skel = []
    skel_xtremes = []
    for i in range(len(xmls)):
        # if father_folder in xmls[i]:
        xmls_annotator = xmls[i]
        try:
            XY1 = xml_points(xmls_annotator)[nn-1]
        except:
            print()

        if class_img == 1:
            skel.append(XY1)

        if class_img == 2:
            points_xtremes = np.concatenate((XY1[0:ptsN, :], XY1[len(XY1) - ptsN:len(XY1), :]), axis=0)
            points = XY1[ptsN:len(XY1) - ptsN, :]

            skel.append(points)
            skel_xtremes.append(points_xtremes)

    if class_img == 1:
        img_skl = build_skel(1944, 1944, skel)
        img_skl = rz_norm_img(img_skl, 255)

    if class_img == 2:
        imgskl = build_skel(1944, 1944, skel)
        skl_ends = build_skel(1944, 1944, skel_xtremes)

        zrs = np.ones((1944, 1944))
        RGB = np.zeros((1944, 1944, 3))
        RGB[:, :, 0] = (zrs - (imgskl + skl_ends)) * 255
        RGB[:, :, 1] = skl_ends * 255
        RGB[:, :, 2] = imgskl * 255

        skl_mask = ((imgskl + skl_ends)>0)*255
        skl_mask = skl_mask.astype(np.uint8)
    return img, RGB, skl_mask


def int2pil(mask_true):
    R = (mask_true == 0) * 255
    G = (mask_true == 1) * 255
    B = (mask_true == 2) * 255

    sz_ = R.shape
    RGB = np.zeros((sz_[0], sz_[1], 3))
    RGB[:, :, 0] = R
    RGB[:, :, 1] = G
    RGB[:, :, 2] = B

    return Image.fromarray(np.uint8(RGB))


def img2parts(img, dim, p):
    n = int(dim[0] / p)

    sz = img.shape
    if len(sz) < 3:
        frames = np.empty((n, n, p * p))
        cnt = 0
        for j in range(p):
            for i in range(p):
                frames[:, :, cnt] = img[i * n:(i + 1) * n, j * n:(j + 1) * n]
                cnt = cnt + 1
    else:
        frames = np.empty((n, n, sz[2], p * p))
        cnt = 0
        for j in range(p):
            for i in range(p):
                frames[:, :, :, cnt] = img[i * n:(i + 1) * n, j * n:(j + 1) * n, :]
                cnt = cnt + 1

    return frames


def get_true_labels(img, mask, n_label):

    v_label = []
    for i in range(mask.shape[2]):
        mask_temp = mask[:, :, i]

        if mask_temp.max() > n_label:
            v_label.append(i)

    img_new = np.zeros((img.shape[0], img.shape[1], len(v_label)))
    mask_new = np.zeros((img.shape[0], img.shape[1], len(v_label)))

    for i in range(len(v_label)):
        img_new[:, :, 0] = img[:, :, v_label[i]]
        mask_new[:, :, 0] = mask[:, :, v_label[i]]

    return img_new, mask_new


def decodec_img3(imgs):
    try:
        trans = transforms.ToPILImage()
        predict_mask = trans(imgs)
    except:
        predict_mask = imgs

    # predict_mask.show()
    pix = predict_mask.load()
    h, w = predict_mask.size
    img_dec = np.zeros((h, w), np.int8)

    for i in range(h):
        for j in range(w):
            dec = list(pix[i, j])
            dec[0] = dec[0] - 5
            v = dec.index(max(dec))
            img_dec[j, i] = v

    return img_dec


def dec_iou(img_dec, true_masks):
    IoU_dec = []
    for i in range(9):
        C0_dec = ((img_dec == i) * 255)
        C0_mask = ((true_masks == i) * 255)
        AND = cv2.bitwise_and(C0_dec, C0_mask)
        OR = cv2.bitwise_or(C0_dec, C0_mask)
        C0_and = np.count_nonzero(AND == 255)
        C0_or = np.count_nonzero(OR == 255)
        if C0_or != 0:
            IoU_dec.append(C0_and / C0_or)
        else:
            IoU_dec.append(0)
    return IoU_dec


def detect_worm(img_RGB, nc):
    BW_img = (img_RGB > 1) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    BWD_img = cv2.dilate(BW_img.astype(np.uint8), kernel, iterations=1)
    labels = label(cv2.bitwise_not(BWD_img), background=1)  # same image_binary as above
    BW_img = np.zeros((img_RGB.shape[0], img_RGB.shape[1])).astype(np.uint8)
    for i in range(2, labels.max()):
        # i_img = cv2.bitwise_and((labels == i)*255, BW_img)
        i_img = ((labels == i)*255).astype(np.uint8)
        dec_vec = []
        for t in range(1, int(img_RGB.max())):
            AND = ((img_RGB == t) * 255).astype(np.uint8)
            AND = cv2.bitwise_and(AND, i_img)
            vt = np.count_nonzero(AND > 0)
            if vt > 1:
                dec_vec.append(1)
            else:
                dec_vec.append(0)
        vt = sum(dec_vec)
        if vt > nc:
            BW_img = cv2.bitwise_or(BW_img, i_img)

    BW_img = cv2.bitwise_and(((img_RGB > 1) * 255).astype(np.uint8), BW_img)
    return BW_img


def iou_img(img_dec, true_masks, wd):
    IoU_dec = 0

    img_dec = img_dec.astype(np.uint8)
    true_masks = true_masks.astype(np.uint8)

    if wd == 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img_dec = cv2.dilate(img_dec, kernel, iterations=1)
        true_masks = cv2.dilate(true_masks, kernel, iterations=1)

    AND = cv2.bitwise_and(img_dec, true_masks)
    OR = cv2.bitwise_or(img_dec, true_masks)
    C0_and = np.count_nonzero(AND > 0)
    C0_or = np.count_nonzero(OR > 0)
    if C0_or != 0:
        IoU_dec = C0_and / C0_or

    return IoU_dec


def true_worm(true_masks, predict_mask):

    G_true = cv2.bitwise_or(((true_masks == 1) * 255).astype(np.uint8), ((true_masks == 2) * 255).astype(np.uint8))
    G_predict = cv2.bitwise_or((predict_mask == 1) * 255, (predict_mask == 2) * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    G_predictD = cv2.dilate(G_predict.astype(np.uint8), kernel, iterations=1)

    labels = label(cv2.bitwise_not(G_predictD), background=1)  # same image_binary as above
    BW_img = np.zeros((G_predict.shape[0], G_predict.shape[1])).astype(np.uint8)
    for i in range(2, labels.max()):
        i_img = ((labels == i)*255).astype(np.uint8)
        AND = cv2.bitwise_and(G_true, i_img)
        vt = np.count_nonzero(AND > 0)
        if vt > 1:
            BW_img = cv2.bitwise_or(BW_img, i_img)

    return cv2.bitwise_and(G_predict.astype(np.uint8), BW_img)
