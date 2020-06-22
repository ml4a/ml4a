
class EasyDict(dict):
    
    def __init__(self, *args, **kwargs):
        super(EasyDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
    
def display_local_videos(videos):
    videos = videos if instance(videos, list) else [videos]
    html_str = ''
    for video in videos:
        html_str += '<video controls style="margin:2px" src="%s"></video>' % video
    IPython.core.display.display(ipy.HTML(html_str))    
