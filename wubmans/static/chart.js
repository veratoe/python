var createChart = function () {

    var labels = results.map(function(item, index) {
       return index; 
    });

    var data = results.map(function(item) {
        return item.loss;
    });

    new Chart(document.getElementById("loss_chart").getContext("2d"), {

        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: "rgb(254, 99, 132)",
                borderColor: "none",
                fill: false
            }]
        },
        options: { 
            legend: {
                display: false
            },
            responsive: false,
            scales: {
                xAxes: [{
                    gridLines: {
                        display: false
                    },
                    ticks: {
                        padding: 19,
                        //display: false,
                        maxTicksLimit: 9,
                    }
                }],
                yAxes: [{
                    gridLines: {
                        display: false
                    },
                    ticks: {
                        maxTicksLimit: 4 
                    }
                }]
            },
            onClick: function (a, b) { 
                console.log(b[-1])
                if (!b || typeof b !== "object") return;
                var $blob = $(".blob#" + (b[-1]._index));
                $("html, body").animate({ scrollTop: $blob.offset().top - 99 })
                $(".blob").removeClass("active");
                $blob.addClass("active");

            }
        }

    });

}
