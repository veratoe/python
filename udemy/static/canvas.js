var c = document.querySelector("canvas");
var ctx = c.getContext('2d');

ctx.lineWidth = 5;
ctx.lineJoin = 'round';
ctx.lineCap = 'round';
ctx.StrokeStyle = '#777';

var mouse = {x: 0, y: 0}

c.addEventListener('mousemove', function(e) {
    mouse.x = e.pageX - this.offsetLeft;
    mouse.y = e.pageY - this.offsetTop;
});

c.addEventListener('mousedown', function() {
    ctx.beginPath();
    ctx.moveTo(mouse.x, mouse.y);
    c.addEventListener('mousemove', paint)
});

c.addEventListener('mouseup', function() {
    c.removeEventListener('mousemove', paint)
});

var paint = function () {
    ctx.lineTo(mouse.x, mouse.y);
    ctx.stroke();
}

var b = document.querySelector('button');
b.addEventListener('click', () => {
    var data = c.toDataURL("image/png");
    console.log(data);
});

var r = document.querySelector("#reset");
r.addEventListener("click", () => {
    ctx.clearRect(0, 0, c.width, c.height);
});
