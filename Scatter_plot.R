library('viridis', quietly = T)
library('ggpointdensity', quietly = T)
library('tidyverse', quietly = T)
library('signs', quietly = T)

custom.breaks <- function(bmin, bmax, digits = 0, length.out = 8, zero = TRUE) {
	bmin = floor(bmin * (10 ^ digits)) / (10 ^ digits)
	bmax = ceiling(bmax * (10 ^ digits)) / (10 ^ digits)
	if (bmin > 0 | bmax < 0) zero = FALSE
	d = round((bmax - bmin) / (length.out - 1), digits)
	if (d == 0) d = round((bmax - bmin) / (length.out - 1), digits+1)
	if (zero) {
		return(unique(c(rev(seq(0, bmin, -d)), seq(0, bmax, d))))
	} else {
		return(seq(bmin, bmax, d))
	}
}

args <- commandArgs(trailingOnly=TRUE)
f.yval = args
if (endsWith(f.yval, 'csv')){
	pred <- read.csv(f.yval)
} else {pred <- read.table(f.yval, header = T, stringsAsFactors = F)}

xy_min = min(pred %>% select(y, y_pred))
xy_max = max(pred %>% select(y, y_pred))
xy_pad = (xy_max - xy_min) * 0.05
xybreaks = custom.breaks(xy_min, xy_max)
rs = cor(pred %>% select(y, y_pred), method = 's')[1,2] 
rp = cor(pred %>% select(y, y_pred), method = 'p')[1,2] 

mag = 1
line_w = 0.75 / .pt * mag
font_size = 6 * mag
gplot <- ggplot(pred, aes(x = y, y = y_pred)) + 
	geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = 'gray20', size = line_w) +
	geom_pointdensity(adjust = 5, size = 0.1) +
	scale_color_viridis(option="magma") +
	labs(x = 'Measured tail length change (nt)', y = 'Predicted tail length change (nt)') +
	coord_cartesian(xlim = c(xy_min - xy_pad, xy_max + xy_pad), ylim = c(xy_min - xy_pad, xy_max + xy_pad), expand = F, clip = 'off') +
	#coord_cartesian(xlim = c(-5, 20), ylim = c(-5, 20), expand = F, clip = 'off') +
	scale_x_continuous(breaks = xybreaks, labels = signs(xybreaks, accuracy = 1)) +
	scale_y_continuous(breaks = xybreaks, labels = signs(xybreaks, accuracy = 1)) + 
	annotate('text', x = xy_min, y = xy_max, label = deparse(bquote('n ='~.(nrow(pred)))), parse = T, hjust = 0, size = font_size * 5 / 14) +
	annotate('text', x = xy_min, y = xy_max - (xy_max - xy_min) * 0.07, label = deparse(bquote(italic('R')[s]~'='~.(format(rs, digits = 2)))), parse = T, hjust = 0, size = font_size * 5 / 14) +
	annotate('text', x = xy_min, y = xy_max - (xy_max - xy_min) * 0.14, label = deparse(bquote(italic('R')[p]~'='~.(format(rp, digits = 2)))), parse = T, hjust = 0, size = font_size * 5 / 14) +
  	theme(	
		plot.background = element_rect(fill = 'transparent',color=NA),
		plot.title = element_text(size = 10, face = 'bold', hjust =0.5, margin=margin(t=0,r=0,b=1,l=0)),
		plot.margin =unit(c(0.01,0.01,0.01,0.01), "inch"),
		panel.background = element_blank(),
		panel.grid = element_blank(),
		
		axis.text.y = element_text(size=font_size,color='black'),
		axis.text.x = element_text(hjust = 0.5, size=font_size, color='black'),
		axis.line.x = element_line(color='black',size=line_w),
		axis.line.y = element_line(color='black',size=line_w),
		axis.title.x = element_text(size=font_size,color='black', vjust = 1, margin=margin(1,0,0,0)),
		axis.title.y = element_text(size=font_size,color='black', vjust = 0.2),
		axis.ticks.x = element_line(color = "black", size = line_w),
		axis.ticks.y = element_line(color = "black", size = line_w),

		legend.position = 'none',
		legend.title = element_blank(),
		legend.background = element_blank(),
		legend.text = element_text(size=font_size,color='black'),
		legend.key.height = unit(0.1, 'inch'),
		legend.key.width = unit(0.1, 'inch'),
		legend.key = element_blank(),
		legend.box.margin = margin(0,0,0,0),
		legend.margin = margin(0,0,0,0),

		aspect.ratio = 1
		)
fname <- paste0(unlist(strsplit(f.yval, '.csv'))[1], '_plot_predciton.png')
png(file=fname, width=2*mag, height=2*mag,unit='in',res=300,bg='transparent')
#gt <- ggplot_gtable(ggplot_build(gplot))
print(gplot)
dev.off()

