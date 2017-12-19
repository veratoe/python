var c = $("canvas")[0];
var ctx = c.getContext('2d');
var imageData = ctx.getImageData(0, 0, c.width, c.height);

var weights;

var pixelSize = 15;
var kernel_size = 5;

setInterval(() => {

    $.getJSON('weights', function(data) {

        try {
            weights = JSON.parse(data)['weights'];
        }

        catch(e) {
        }

        finally {
            self.status_pending = false;
        }

        drawWeights();

    });

}, 200);

var drawWeights = function() {

    var show_greyscale = $("input#show_greyscale").is(":checked");

    for (var f = 0; f < weights.length; f++) {

        var offsetX = f % 8;
        var offsetY = Math.floor (f / 8);

        for (var x = 0; x < kernel_size; x++) {
            for (var y = 0; y < kernel_size; y++) {
                for (var px = 0; px < pixelSize; px++){
                    for (var py = 0; py < pixelSize; py++){
                        var o = ((px + x * pixelSize + (offsetX * pixelSize * 6)) + ((py + y * pixelSize + (offsetY * pixelSize * 6)) * c.width)) * 4;
                        var g = 0.21 * weights[f][x][y][0] + 0.72 * weights[f][x][y][1] + 0.07 * weights[f][x][y][2];
                        imageData.data[o]     = (show_greyscale ? g : weights[f][x][y][0]) * 128 + 128;
                        imageData.data[o + 1] = (show_greyscale ? g : weights[f][x][y][1]) * 128 + 128;
                        imageData.data[o + 2] = (show_greyscale ? g : weights[f][x][y][2]) * 128 + 128;
                        imageData.data[o + 3] = 255;
                    }
                }
            }
        }
    }

    ctx.putImageData(imageData, 0, 0);

}
