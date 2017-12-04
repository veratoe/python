new Vue({ 
    el: "#vue-app",
    data: {
        status: {},
        status_pending: null,
        results: [],

        seed: null,
        generate_status: null,
        generated_text: null,

        temperature: 0.1,
        temperatures: [],
        show_certainties: null,

    },

    mounted() {

        var self = this;

        setInterval(function () {

            if (self.status_pending) {
                
                return;
            }

            self.status_pending = true;

            $.getJSON('status', function(data) { 

                try {
                    self.status = JSON.parse(data);
                }

                catch(e) {
                }

                finally {
                    self.status_pending = false;
                }

                status_pending = false;

            })
            
        }, 1000)


        this.fetchResults();

    },

    watch: {

        'results' () {
            createChart(this.results)
            this.results.forEach((element, index) => { this.temperatures[index] = 0; })
        }
    },

    methods: {

        fetchResults () {

            var self = this;

            $.getJSON('results', function(data) { 
                r = data.split("#@#");
                r = r.map(function(json) { 
                    try {
                        var j = JSON.parse(json);
                        return j;
                    }
                    catch(e) {
                    }

                });

                r = r.filter(function (item) { return typeof item === "object" });

                self.results = r;
            })


        },

        renderCertainties (text) {

            var c1 = "#e6194b";
            var c2 = "#3cb44b";


            // Converts a #ffffff hex string into an [r,g,b] array
            var h2r = function(hex) {
                var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
                return result ? [
                    parseInt(result[1], 16),
                    parseInt(result[2], 16),
                    parseInt(result[3], 16)
                ] : null;
            };
            
            // Inverse of the above
            var r2h = function(rgb) {
                return "#" + ((1 << 24) + (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]).toString(16).slice(1);
            };

            var interpolate = (a, b, i) => { 
                a = h2r(a)
                b = h2r(b)

                var r = a[0] + ((b[0] - a[0]) * i | 0)
                var g = a[1] + ((b[1] - a[1]) * i | 0)
                var b = a[2] + ((b[2] - a[2]) * i | 0)

                return r2h([r, g, b]);
            }

            var html = "";
            for (c in text.result_text) {
                if (text.result_text[c] !== " ") {
                    html += "<span title='" + text.certainties[c] + "' style='cursor: pointer; color: " + interpolate(c1, c2, text.certainties[c]) + "'>" + text.result_text[c] + "</span>" 
                } else {
                    html += text.result_text[c];
                }
            }

            return html;

        },

        generate () {

            var self = this;

            this.generate_status = "PENDING";

            $.ajax({
                url: "/seed", 
                type: "POST",
                data: JSON.stringify({ "seed": self.seed.toLowerCase(), "temperature": self.temperature }),
                contentType: 'application/json; charset=utf-8',
                dataType: 'json',
                success: function (data) {
                    self.generate_status = "DONE";
                    self.generated_text = data;
                }

            })
        },

    },

    filters: {

        parse_seconds (value) {

            var minutes = Math.floor(Number(value) / 60)
            var seconds = Math.floor(Number(value) % 60)
            return minutes + ":" + (seconds < 10 ? "0" : "") + seconds;

        },

        parse_float (value) {
            return Number(value).toFixed(4);
        },


    }

});
