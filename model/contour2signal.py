from madule.image_madule import *
from model.contour_finder import ContourFinder


def contour2signal(path, a):
    cnt_finder = ContourFinder(path)
    cnt_list = cnt_finder.draw_contours()
    image = read_img_clahe(path)
    signal_list = []

    for i in np.array(cnt_list):
        img = image[i[2]:i[2] + i[4], i[1]:i[1] + i[3]]
        signal = image2signal(img)
        signal_list.append(signal)

    df = pd.DataFrame(signal_list)
    df.to_csv(f"files/csv_files/signal{a}.csv")
