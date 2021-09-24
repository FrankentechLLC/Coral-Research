"""
Plotting

Creates an interactive plot of papers grouped by relatedness.

- Requires clustering labels and a dataset of papers 
reduced to two dimensions. 
- Topic models each paper cluster to reveal its key terms.

Bokeh pairs the actual papers with their positions on the t-SNE 
plot, showing how papers fit together to enable dataset exploration
and clustering evaluation.
"""

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import os
from lib.plot_text import (
    header, 
    description, 
    description2, 
    cite, 
    description_search, 
    description_slider, 
    notes, 
    dataset_description, 
    toolbox_header
)
from lib.call_backs import input_callback, selected_code
import bokeh
from bokeh.models import (
    ColumnDataSource, 
    HoverTool, 
    LinearColorMapper, 
    CustomJS, 
    Slider, 
    TapTool, 
    TextInput
)
from bokeh.palettes import Category20
from bokeh.transform import linear_cmap, transform
from bokeh.io import output_file, show, output_notebook
from bokeh.plotting import figure
from bokeh.models import RadioButtonGroup, TextInput, Div, Paragraph
from bokeh.layouts import column, widgetbox, row, layout
from bokeh.layouts import column


def cluster_plot(embedded_text, clusters, k):
    """Return and show a plot of k-means clusters."""
    get_ipython().run_line_magic('matplotlib', 'inline')
    sns.set(rc={'figure.figsize':(15, 15)})
    palette = sns.hls_palette(k, l = .4, s = .9)
    sns.scatterplot(
        embedded_text[:,0], 
        embedded_text[:,1], 
        hue = clusters, 
        legend = 'full', 
        palette = palette
    )
    plt.title('t-SNE with Kmeans Labels')
    plt.savefig("plots/improved_cluster_tsne.png")
    plt.show()


# ## Load the Keywords per Cluster
def interactive_plot(
    dataframe, 
    embedded_text, 
    clusters,
    random_state = 42,
    plot_width = 1280,
    plot_height = 850
):
    topic_path = os.path.join(os.getcwd(), 'lib', 'topics.txt')
    with open(topic_path) as f:
        topics = f.readlines()

    # show on notebook
    output_notebook()
    # target labels
    y_labels = clusters

    # data sources
    source = ColumnDataSource(data=dict(
        x = embedded_text[:,0], 
        y = embedded_text[:,1],
        x_backup = embedded_text[:,0],
        y_backup = embedded_text[:,1],
        desc = y_labels, 
        titles= dataframe['title'],
        authors = dataframe['authors'],
        journal = dataframe['journal'],
        abstract = dataframe['abstract_summary'],
        labels = ["C-" + str(x) for x in y_labels],
        links = dataframe['doi']    
    ))

    # hover over information
    hover = HoverTool(tooltips=[
        ("Title", "@titles{safe}"),
        ("Author(s)", "@authors{safe}"),
        ("Journal", "@journal"),
        ("Abstract", "@abstract{safe}"),
        ("Link", "@links")
    ],
    point_policy = "follow_mouse")

    # map colors
    initial_palette = Category20[20]
    random.Random(random_state).shuffle(initial_palette)

    mapper = linear_cmap(
        field_name = 'desc', 
        palette = Category20[20],
        low = min(y_labels),
        high = max(y_labels)
    )

    # prepare the figure
    plot = figure(
        plot_width = plot_width, 
        plot_height = plot_height, 
        tools = [
            hover, 
            'pan', 
            'wheel_zoom', 
            'box_zoom', 
            'reset', 
            'save', 
            'tap'
        ], 
        title = "Clustering of the literature with t-SNE and K-Means", 
        toolbar_location = "above"
    )

    # plot settings
    plot.scatter(
        'x', 
        'y', 
        size = 5, 
        source = source,
        fill_color = mapper,
        line_alpha = 0.3,
        line_width = 1.1,
        line_color = "black",
        legend = 'labels'
    )
    plot.legend.background_fill_alpha = 0.6


    # Widgets

    # Keywords
    text_banner = Paragraph(
        text= 'Keywords: Slide to specific cluster to see the keywords.', 
        height = 25
    )
    input_callback_1 = input_callback(plot, source, text_banner, topics)

    # currently selected article
    div_curr = Div(
        text = """Click on a plot to see the link to the article.""",
        height = 150
    )
    callback_selected = CustomJS(
        args = dict(source=source, current_selection=div_curr), 
        code = selected_code()
    )
    taptool = plot.select(type=TapTool)
    taptool.callback = callback_selected

    # WIDGETS
    slider = Slider(
        start = 0, 
        end = 20, 
        value = 20, 
        step = 1, 
        title = "Cluster #", 
        callback = input_callback_1
    )
    keyword = TextInput(title="Search:", callback=input_callback_1)

    # pass call back arguments
    input_callback_1.args["text"] = keyword
    input_callback_1.args["slider"] = slider

    header.sizing_mode = "stretch_width"
    header.style = {
        'color': '#2e484c', 
        'font-family': 'Julius Sans One, sans-serif;'
    }
    header.margin = 5

    description.style = {
        'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;',
        'font-size': '1.1em'
    }
    description.sizing_mode = "stretch_width"
    description.margin = 5

    description2.sizing_mode = "stretch_width"
    description2.style = {
        'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 
        'font-size': '1.1em'
    }
    description2.margin = 10

    description_slider.style = {
        'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 
        'font-size': '1.1em'
    }
    description_slider.sizing_mode = "stretch_width"

    description_search.style = {
        'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 
        'font-size': '1.1em'
    }
    description_search.sizing_mode = "stretch_width"
    description_search.margin = 5

    slider.sizing_mode = "stretch_width"
    slider.margin = 15

    keyword.sizing_mode = "scale_both"
    keyword.margin = 15

    div_curr.style = {
        'color': '#BF0A30', 
        'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 
        'font-size': '1.1em'
    }
    div_curr.sizing_mode = "scale_both"
    div_curr.margin = 20

    text_banner.style = {
        'color': '#0269A4', 
        'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 
        'font-size': '1.1em'
    }
    text_banner.sizing_mode = "scale_both"
    text_banner.margin = 20

    plot.sizing_mode = "scale_both"
    plot.margin = 5

    dataset_description.sizing_mode = "stretch_width"
    dataset_description.style = {
        'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 
        'font-size': '1.1em'
    }
    dataset_description.margin = 10

    notes.sizing_mode = "stretch_width"
    notes.style ={
        'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 
        'font-size': '1.1em'
    }
    notes.margin = 10

    cite.sizing_mode = "stretch_width"
    cite.style = {
        'font-family': 'Helvetica Neue, Helvetica, Arial, sans-serif;', 
        'font-size': '1.1em'
    }
    cite.margin = 10

    r = row(div_curr,text_banner)
    r.sizing_mode = "stretch_width"

    # layout
    l = layout([
        [header],
        [description],
        [description_slider, description_search],
        [slider, keyword],
        [text_banner],
        [div_curr],
        [plot],
        [description2, dataset_description, notes, cite],
    ])
    l.sizing_mode = "scale_both"


    # show
    output_file('plots/interactive_plot.html')
    show(l)