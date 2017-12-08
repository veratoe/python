var c = document.querySelector("canvas");
var ctx = c.getContext('2d');

ctx.lineWidth = 25;
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.strokeStyle = 'rgba(0, 0, 0, 0.04)';

var mouse = {x: 0, y: 0}

$(c).on('mousemove', function(e) {
    mouse.x = e.pageX - this.offsetLeft;
    mouse.y = e.pageY - this.offsetTop;
});

$(c).on('mousedown', function() {
    ctx.beginPath();
    ctx.moveTo(mouse.x, mouse.y);
    c.addEventListener('mousemove', paint)
});

$(c).on('mouseup', function() {
    c.removeEventListener('mousemove', paint)
});

var paint = function () {
    ctx.lineTo(mouse.x, mouse.y);
    ctx.stroke();
}

$("#submit").on('click', () => {
    var data = c.toDataURL("image/png");
    $.ajax({
        type: "POST",
        url: "/upload",
        data: {
            url: data
        }
    }).done((response) => {
        var sorted = response.slice().sort((a, b) => { return b - a });
        console.log(sorted);

        var text = "";

        text += "<p>Het kan een " + response.indexOf(sorted[0]) + " zijn, s => " + sorted[0] + "</p>";
        text += "<p>Het kan ook een " + response.indexOf(sorted[1]) + " zijn, s => " + sorted[1] + "</p>";

        $("#results").html(text);
    });

});

$("#reset").on("click", () => {
    ctx.clearRect(0, 0, c.width, c.height);
});
