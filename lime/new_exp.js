var RuleExplanation = function(svg, text, pred, rules) {
  var width = parseInt(svg.style('width'));
  var height = parseInt(svg.style('height'));
  var padding_right = 100;
  var padding_left = 50;
  var padding_bottom = 50;
  var xscale = d3.scale.linear([0,1]).range([padding_left, width - padding_right])
  var x_axis = d3.svg.axis().scale(xscale).orient('bottom');
  svg.append('g')
     .style('fill', 'none')
     .style('stroke', 'black')
     .style('shape-rendering', 'crispEdges')
     .style('font-family', 'sans-serif')
     .style('font-size', '11px')
     .attr('transform', 'translate(0, ' + (height - padding_bottom) + ')')
     .call(x_axis)
  svg.append('ellipse')
     .attr('cx', xscale(pred))
     .attr('cy', (height - padding_bottom))
     .attr('rx', 2)
     .attr('ry', 10)
     .style('fill', 'red');
  svg.append('text')
     .attr('x', xscale(pred))
     .attr('y', (height - padding_bottom - 15))
     .attr('text-anchor', 'middle')
     .text('Prediction')
  svg.append('text')
     .attr('x', xscale(0.5))
     .attr('y', 20)
     .attr('text-anchor', 'middle')
     .text(text);
     
  var space = {}
  _.forEach(rules, function(rule) {
    var radius =  xscale(rule.spread) - xscale(0)
    var center = xscale(rule.pred)
    var start = center - radius
    var end = center + radius
    var level = findLevel(start - 20, end + 20, space);
    //console.log(level);
    if (space[level] === undefined) {
      space[level] = []
    }
    space[level].push([start - 20, end + 20])
    //console.log(space);
    svg.append('ellipse')
       .attr('cx', center)
       .attr('cy', (height - padding_bottom))
       .attr('rx', radius)
       .attr('ry', 30)
       .style('fill', 'blue')
       .style('fill-opacity', .5);
    var i = 0;
    _.forEach(rule.present, function(x) {
      svg.append('text')
         .attr('x', center)
         .attr('y', (height - padding_bottom - 35 - i * 15 - level * 50))
         .attr('text-anchor', 'middle')
         .style('fill', 'green')
         .text(x);
      i += 1;
    })
    _.forEach(rule.absent, function(x) {
      svg.append('text')
         .attr('x', xscale(rule.pred))
         .attr('y', (height - padding_bottom - 35 - i * 15))
         .attr('text-anchor', 'middle')
         .style('fill', 'red')
         .style('text-decoration', 'line-through')
         .text(x);
      i += 1;
    })
  });
  // svg.append('line')
  //   .attr('x1', xscale(0))
  //   .attr('x2', xscale(1))
  //   .attr('y1', height - 20)
  //   .attr('y2', height - 20)
  //   .style('stroke-width', 2)
  //   .style('stroke', 'black');
}

var findLevel = function(start, end, space) {
  var max_level = -1;
  for (var level in Object.keys(space)) {
    console.log("OI" + level);
    max_level = Math.max(max_level, level)
    var stop = true;
    for (var i = 0; i < space[level].length; ++i) {
      var limits = space[level][i];
      console.log(limits);
      if ((start > limits[0] && start < limits[1]) || (end > limits[0] && end < limits[1])) {
        stop = false;
      }
    }
    if (stop) {
      return level;
    }
  }
  return max_level + 1;
}


// var svg = d3.select('body').append('svg').style('width', '100%').style('height', '200px');
// rules = [{'pred': 0.84206239571580754, 'spread': 0.088147512573176073, 'present': [], 'absent': ['not', 'bad']}, {'pred': 0.0027110360138727671, 'spread': 0.00092662877643485953, 'present': ['bad'], 'absent': ['not']}, {'pred': 0.045516806883085979, 'spread': 0.01450470709476792, 'present': ['not'], 'absent': ['bad']}, {'pred': 0.93489980006824791, 'spread': 0.039745273931326604, 'present': ['not', 'bad'], 'absent': []}]
// text=' This is not bad .'
// pred= 0.975625471758
// RuleExplanation(svg, text, pred, rules);
