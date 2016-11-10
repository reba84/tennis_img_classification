import os
import sys
import urllib
import text_file_formatter as tff

def get_image_links(image_number):
    '''
    This function takes a list of image numbers and adds them to a base url
    and returns the full url that can be used to download images
    '''
    base_url = "url_here"
    url = '{}{}'.format(base_url, image_number)
    return url

def download_image(link, filename):
    '''
    takes a url link and a name a new local filename for the image to be saved under
    '''
    urllib.urlretrieve(link, filename)

if __name__ == '__main__':

    tup_lst = tff.file_reader('text_file_name')  ## use function from text_file_formatter.py
    print tup_lst[0], tup_lst[1]  ## checks for list of links and labels
    serve, InsidePoint, OutsidePoint = tff.label_split(tup_lst)  ## use other function from text_file_formatter.py
    print len(serve), len(InsidePoint), len(OutsidePoint) ## checks the length of lists for total samples

    '''
    these loops are for downloading all the images into files with the correct labels
    they get the link from the link list from the text_file_formatter label_split and
    the label then save the file in the folder the script is running from.

    '''
    counter = 1 #keeps track of download
    # for item in serve:
    #     link = get_image_links(item[0])
    #     output_string = item[1]+item[0]
    #     print "downloading", counter
    #     counter += 1
    #     #print link, output_string
    #     download_image(link, output_string)

    # for item in InsidePoint:
    #     link = get_image_links(item[0])
    #     output_string = item[1]+item[0]
    #     print "downloading", counter
    #     counter += 1
    #     #print link, output_string
    #     download_image(link, output_string)
    #
    # for item in OutsidePoint:
    #     link = get_image_links(item[0])
    #     output_string = item[1]+item[0]
    #     print "downloading", counter
    #     counter += 1
    #     print link, output_string
    #     download_image(link, output_string)
