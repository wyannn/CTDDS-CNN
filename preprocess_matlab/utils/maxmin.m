function  res = maxmin(data, ymin, ymax)

xmax = max(data(:));
xmin = min(data(:));
res = (ymax-ymin) * (data - xmin) / (xmax - xmin) + ymin;

end