var Explanation = function(class_names) {
  this.names = class_names;
  this.names.push('Other');
  if (class_names.length < 10) {
    this.colors = d3.scale.category10().domain(this.names);
    this.colors_i = d3.scale.category10().domain(_.range(this.names.length));
  }
  else {
    this.colors = d3.scale.category20().domain(this.names);
    this.colors_i = d3.scale.category20().domain(_.range(this.names.length));
  }
}
Explanation.prototype.PredictProba = function(svg, predict_proba) {
  svg.style('float', 'left')
     .style('width', 225);
  this.bar_x = 100;
  this.bar_height = 17;
  this.space_between_bars = 5;
  this.width = 225;
  this.bar_width = this.width - this.bar_x - 31;
  this.x_scale = d3.scale.linear().range([0, this.bar_width]);
  svg.append('text')
     .text('Prediction probabilities')
     .attr('x', 40)
     .attr('y', 20);
  this.bar_yshift=35;
  var mapped = this.MapClasses(this.names, predict_proba);
  var names = mapped[0];
  var data = mapped[1];
  var n_bars = Math.min(5, names.length);
  var bar = svg.append("g");
  var this_object = this;
  for (i = 0; i < data.length; i++) {
    var color = this.colors(names[i]);
    if (names[i] == 'Other' && this.names.length > 20) {
        color = '#5F9EA0';
    }
    rect = bar.append("rect");
    rect.attr("x", this.bar_x)
        .attr("y", this.BarY(i))
        .attr("height", this.bar_height)
        .attr("width", this.x_scale(data[i]))
        .style("fill", color);
    bar.append("rect").attr("x", this.bar_x)
        .attr("y", this.BarY(i))
        .attr("height", this.bar_height)
        .attr("width", this.bar_width - 1)
        .attr("fill-opacity", 0)
        .attr("stroke", "black");
    text = bar.append("text");
    text.classed("prob_text", true);
    text.attr("y", this.BarY(i) + this.bar_height - 3).attr("fill", "black").style("font", "14px tahoma, sans-serif");
    text = bar.append("text");
    text.attr("x", this.bar_x + this.x_scale(data[i]) + 5)
        .attr("y", this.BarY(i) + this.bar_height - 3)
        .attr("fill", "black")
        .style("font", "14px tahoma, sans-serif")
        .text(data[i].toFixed(2));
    text = bar.append("text");
    text.attr("x", this.bar_x - 10)
        .attr("y", this.BarY(i) + this.bar_height - 3)
        .attr("fill", "black")
        .attr("text-anchor", "end")
        .style("font", "14px tahoma, sans-serif")
        .text(names[i]);
  }
}

Explanation.prototype.BarY = function(i) {
  return (this.bar_height + this.space_between_bars) * i + this.bar_yshift;
}

Explanation.prototype.MapClasses = function(class_names, predict_proba) {
  if (class_names.length <= 6) {
    return [class_names, predict_proba];
  }
  class_dict = _.map(_.range(predict_proba.length), function (i) {return {'name': class_names[i], 'prob': predict_proba[i], 'i' : i};});
  sorted = _.sortBy(class_dict, function (d) {return -d.prob});
  other = new Set();
  _.forEach(_.range(4, sorted.length), function(d) {other.add(sorted[d].name);});
  other_prob = 0;
  ret_probs = [];
  ret_names = [];
  for (d = 0 ; d < sorted.length; d++) {
    if (other.has(sorted[d].name)) {
      other_prob += sorted[d].prob;
    }
    else {
      ret_probs.push(sorted[d].prob);
      ret_names.push(sorted[d].name);
    }
  };
  ret_names.push("Other");
  ret_probs.push(other_prob);
  return [ret_names, ret_probs];
}

Explanation.prototype.ExplainFeatures = function(svg, class_id, exp_array, title, show_numbers) {
  var bar_height = 17;
  var yshift = 35;
  var max_weight = _.max(_.map(exp_array, function(d) {return Math.abs(d[1]);}));
  //var bar_width = max_weight > .2 ? 110 : 500;
  var bar_width = 300;
  var xscale = d3.scale.linear()
          .domain([0,1])
          .range([0, bar_width]);
  var width = Math.max(240, (xscale(max_weight) + 32) * 2); //270;
  var x_offset = width / 2;
  var total_height = (bar_height + 10) * exp_array.length;
  svg.style('width', width)
     .style('height', yshift + total_height + 10);
  svg.append('text')
     .text(title)
     .attr('y', 20)
     .attr('x', x_offset)
     .attr('text-anchor', 'middle')
  svg.append('text')
     .text('(' + this.names[class_id] + ')')
     .attr('y', 35)
     .attr('x', x_offset)
     .attr('text-anchor', 'middle')
  var yscale = d3.scale.linear()
          .domain([0, exp_array.length])
          .range([yshift,total_height + yshift]);
  for (var i = 0; i < exp_array.length; ++i) {
    var name = exp_array[i][0];
    var score = exp_array[i][1];
    var size = xscale(Math.abs(score));
    var color;
    if (this.names.length == 3) {
      color = score > 0 ? this.colors_i(class_id) : this.colors_i(1 - class_id);
    }
    else {
      color = score > 0 ? this.colors_i(class_id) : this.colors('Other');
      if (this.names.length > 20 && score < 0) {
        color = '#5F9EA0';
      }
    }
    var bar = svg.append('rect')
                 .attr('height', bar_height)
                 .attr('x', score > 0 ? x_offset : x_offset - size)
                 .attr('y', yscale(i) + bar_height)
                 .attr('width', size)
                 .attr('fill', color);
    var text = svg.append('text')
                  .attr('x', score > 0 ? x_offset - 2 : x_offset + 2)
                  .attr('y', yscale(i) + bar_height + 14)
                  .attr('text-anchor', score > 0 ? 'end' : 'begin')
                  .text(name);
    if (show_numbers) {
      var bartext = svg.append('text')
                     .attr('x', score > 0 ? x_offset + size + 1 : x_offset - size - 1)
                     .attr('text-anchor', score > 0 ? 'begin' : 'end')
                     .attr('y', yscale(i) + 30)
                     .text(score.toFixed(2));
    }
  }
  var line = svg.append("line")
                      .attr("x1", x_offset)
                      .attr("x2", x_offset)
                      .attr("y1", bar_height + yshift)
                      .attr("y2", Math.max(bar_height, -5 + bar_height + yscale(exp_array.length)))
                      .style("stroke-width",2)
                      .style("stroke", "black");
  
}

Explanation.prototype.UpdateColors = function(div, class_id) {
  div.style('width', '550px')
     .style('float', 'left');
  var pos_color = this.colors_i(class_id);
  var neg_color = this.names.length == 3 ? this.colors_i(1 - class_id) : this.colors('Other');
  if (this.names.length >= 20) {
    neg_color = '#5F9EA0';
  }

  div.selectAll('.pos').style('background-color', pos_color);
  div.selectAll('.neg').style('background-color', neg_color);
}
