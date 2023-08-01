from model.contour_finder import ContourFinder

cnt_finder = ContourFinder("files/CV-FB.jpg")
cnt_list = cnt_finder.draw_contours()
