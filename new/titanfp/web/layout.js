import $ from 'jquery';
import debounce from 'lodash-es/debounce';

export const editor_box = [];

const border_px = 16;

const min_width = 640;
const min_height = 320;

export function resize_webtool() {
    const vh = window.innerHeight;
    const vw = window.innerWidth;
    const h = vh - (border_px * 2);
    const w = (vw - (border_px * 3)) / 2;

    $('#webtool').css({
        width: vw + 'px',
        height: vh + 'px',
    });
    $('.left').css({
        width: (w + border_px) + 'px',
        height:  h + 'px',
        left: '0px',
        top: border_px + 'px',
    });
    $('.right').css({
        width: w  + 'px',
        height:  h + 'px',
        left: (w + (border_px * 2)) + 'px',
        top: border_px + 'px',
    });

    if (editor_box.length > 0) {
        editor_box[0].layout({width: w + border_px, height: h});
    }
}

resize_webtool();
$(window).on('resize', debounce(resize_webtool, 100));
