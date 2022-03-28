import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker



def plot_formatter(ax, fig=None,
                   title=None, xlabel=None, ylabel=None,  
                   rotation=(0, 0), 
                   xlabel_size=12, ylabel_size=20, title_size=12,
                   xtick_size=12, ytick_size=12,
                   xlabel_pad=10, ylabel_pad=10,
                   title_position=(0.5, 0.9), xlabel_position=None, ylabel_position=None,
                   xaxis_lwd=1.5, yaxis_lwd=1.5,
                   xaxis_date=False, date_format='%b %Y', date_month_interval=1,
                   show_thousand_x=False, show_thousand_y=False,
                   remove_legend=True):
    
    # Labels
    ax.set_xlabel(xlabel, size=xlabel_size, fontfamily='serif', labelpad=xlabel_pad)
    ax.set_ylabel(ylabel, size=ylabel_size, fontweight='bold', fontfamily='serif', labelpad=ylabel_pad)
    ax.set_title(title, x=title_position[0], y=title_position[1], size=title_size, fontweight='bold', fontfamily='serif')

    # Hide top & right axis
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Axis Linewidth
    ax.spines['left'].set_linewidth(yaxis_lwd)
    ax.spines['bottom'].set_linewidth(xaxis_lwd)

    #Xtick size
    ax.tick_params(axis='x', which='major', labelsize=xtick_size, rotation=rotation[0])
    ax.tick_params(axis='y', which='major', labelsize=ytick_size, rotation=rotation[1])
        
    # Date on xaxis
    if xaxis_date:
        years_fmt = mdates.DateFormatter(date_format)
        months_n = mdates.MonthLocator(interval=date_month_interval)
        months_1 = mdates.MonthLocator()
        ax.xaxis.set_minor_locator(months_1)
        ax.xaxis.set_major_locator(months_n)
        ax.xaxis.set_major_formatter(years_fmt)
    
    # X & Y Labels Positions
    if xlabel_position is not None:
        ax.xaxis.set_label_coords(xlabel_position[0], xlabel_position[1])
    if ylabel_position is not None:
        ax.yaxis.set_label_coords(ylabel_position[0], ylabel_position[1])
    
    # Major Format
    if show_thousand_x:
        ax.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    if show_thousand_y:
        ax.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # legend (Keep it as the last part of the code)
    h, l = None, None
    if remove_legend:
        h, l = ax.get_legend_handles_labels()
        if (len(h) & len(l)) > 0:
            ax.legend().remove()

    return h, l 