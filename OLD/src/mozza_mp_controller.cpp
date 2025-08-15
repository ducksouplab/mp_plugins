#include <gst/gst.h>
#include <iostream>
#include <vector>
#include <string.h>

// Schedule of alpha changes (time in nanoseconds)
struct AlphaChange {
    GstClockTime time;
    gdouble alpha;
};

static GstElement *pipeline = nullptr;
static GstElement *mozza = nullptr; // our mozza_mp instance
static std::vector<AlphaChange> changes;

static void on_pad_added(GstElement *element, GstPad *pad, gpointer data) {
    GstElement *decoder = (GstElement *)data;
    GstPad *sinkpad = gst_element_get_static_pad(decoder, "sink");
    if (sinkpad) {
        if (gst_pad_is_linked(sinkpad) == FALSE) {
            gst_pad_link(pad, sinkpad);
        }
        gst_object_unref(sinkpad);
    }
}

static gboolean update_alpha(gpointer /*user_data*/) {
    if (!pipeline || !mozza) return G_SOURCE_CONTINUE;

    gint64 pos = 0;
    if (gst_element_query_position(pipeline, GST_FORMAT_TIME, &pos)) {
        for (auto it = changes.begin(); it != changes.end();) {
            if (it->time <= (GstClockTime)pos) {
                g_object_set(mozza, "alpha", it->alpha, NULL);
                it = changes.erase(it);
            } else {
                ++it;
            }
        }
    }
    return changes.empty() ? G_SOURCE_REMOVE : G_SOURCE_CONTINUE;
}

static gboolean bus_callback(GstBus *bus, GstMessage *msg, gpointer loop) {
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_main_loop_quit((GMainLoop*)loop);
            break;
        case GST_MESSAGE_ERROR: {
            GError *err = nullptr; gchar *dbg = nullptr;
            gst_message_parse_error(msg, &err, &dbg);
            g_printerr("Error: %s\n", err ? err->message : "(unknown)");
            if (err) g_error_free(err);
            g_free(dbg);
            g_main_loop_quit((GMainLoop*)loop);
            break;
        }
        default: break;
    }
    return TRUE;
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    // Usage identical to original, but element/properties differ:
    //   With explicit pipeline:
    //     argv: <prog> "<pipeline string>" <source> <dfm> <output> <times> <alphas>
    //     (Note: the pipeline string must contain 'mozza_mp name=mozza_mp')
    //
    //   Or auto-build:
    //     argv: <prog> <source> <dfm> <output> <times> <alphas>
    //
    // Examples:
    //   ./mozza_mp_controller \
    //     "filesrc location=in.mp4 ! decodebin ! videoconvert ! video/x-raw,format=RGBA ! \
    //       mozza_mp name=mozza_mp model=/models/face_landmarker.task deform=smile.dfm alpha=0 ! \
    //       videoconvert ! autovideosink" \
    //     in.mp4 smile.dfm out.mp4 0,5,10 0.0,1.0,-0.5
    //
    //   ./mozza_mp_controller in.mp4 smile.dfm out.mp4 0,5,10 0.0,1.0,-0.5
    //
    if (argc < 6 || argc > 7) {
        g_printerr("Usage: %s [<pipeline>] <source> <deformation.dfm> <output> <times> <alphas>\n", argv[0]);
        return -1;
    }

    const gboolean has_pipeline_desc = (argc == 7);
    const gchar *pipeline_desc = has_pipeline_desc ? argv[1] : NULL;
    const gchar *source   = argv[has_pipeline_desc ? 2 : 1];
    const gchar *dfm_file = argv[has_pipeline_desc ? 3 : 2];
    const gchar *output   = argv[has_pipeline_desc ? 4 : 3];
    gchar **times_str     = g_strsplit(argv[has_pipeline_desc ? 5 : 4], ",", -1);
    gchar **alphas_str    = g_strsplit(argv[has_pipeline_desc ? 6 : 5], ",", -1);

    const gint n_times  = g_strv_length(times_str);
    const gint n_alphas = g_strv_length(alphas_str);
    if (n_times != n_alphas) {
        g_printerr("Times and alphas arrays must have the same length\n");
        g_strfreev(times_str); g_strfreev(alphas_str);
        return -1;
    }

    GstElement *src=nullptr, *demuxer=nullptr, *decoder=nullptr;
    GstElement *convert1=nullptr, *capsfilter=nullptr, *convert2=nullptr;
    GstElement *sink=nullptr;

    if (has_pipeline_desc) {
        GError *error = NULL;
        pipeline = gst_parse_launch(pipeline_desc, &error);
        if (error) {
            g_printerr("Error parsing pipeline: %s\n", error->message);
            g_error_free(error);
            g_strfreev(times_str); g_strfreev(alphas_str);
            return -1;
        }
        // IMPORTANT: your pipeline_desc must name the element "mozza_mp"
        mozza = gst_bin_get_by_name(GST_BIN(pipeline), "mozza_mp");
        if (!mozza) {
            g_printerr("Could not find element named 'mozza_mp' in the pipeline string\n");
            g_strfreev(times_str); g_strfreev(alphas_str);
            return -1;
        }
    } else {
        pipeline  = gst_pipeline_new("mozza-mp-pipeline");
        src       = gst_element_factory_make("filesrc", "source");
        demuxer   = gst_element_factory_make("qtdemux", "demuxer");
        decoder   = gst_element_factory_make("avdec_h264", "decoder");
        convert1  = gst_element_factory_make("videoconvert", "convert1");
        capsfilter= gst_element_factory_make("capsfilter", "to_rgba");
        mozza     = gst_element_factory_make("mozza_mp", "mozza_mp");
        convert2  = gst_element_factory_make("videoconvert", "convert2");
        sink      = gst_element_factory_make("filesink", "sink");

        if (!pipeline || !src || !demuxer || !decoder || !convert1 || !capsfilter ||
            !mozza || !convert2 || !sink) {
            g_printerr("Not all elements could be created\n");
            g_strfreev(times_str); g_strfreev(alphas_str);
            return -1;
        }

        // Set properties
        g_object_set(src, "location", source, NULL);
        g_object_set(sink, "location", output, NULL);
        // Set DFM on the element (property name is 'deform', not 'deform-file')
        g_object_set(mozza, "deform", dfm_file, NULL);

        // You MUST also set the MediaPipe .task model path here or via env/pipeline
        // Example model path (change to your actual model):
        // g_object_set(mozza, "model", "/models/face_landmarker.task", NULL);

        // Force RGBA caps before mozza_mp
        GstCaps *caps = gst_caps_from_string("video/x-raw,format=RGBA");
        g_object_set(capsfilter, "caps", caps, NULL);
        gst_caps_unref(caps);

        gst_bin_add_many(GST_BIN(pipeline),
                         src, demuxer, decoder, convert1, capsfilter,
                         mozza, convert2, sink, NULL);

        if (!gst_element_link(src, demuxer)) {
            g_printerr("filesrc ! demuxer link failed\n"); return -1;
        }
        g_signal_connect(demuxer, "pad-added", G_CALLBACK(on_pad_added), decoder);

        if (!gst_element_link_many(decoder, convert1, capsfilter, mozza, convert2, sink, NULL)) {
            g_printerr("Downstream linking failed\n");
            return -1;
        }
    }

    // Prepare scheduled alpha changes (seconds → GstClockTime)
    for (gint i = 0; i < n_times; ++i) {
        GstClockTime when = g_ascii_strtoull(times_str[i], NULL, 10) * GST_SECOND;
        gdouble val = g_ascii_strtod(alphas_str[i], NULL);
        changes.push_back(AlphaChange{ when, val });
    }
    g_strfreev(times_str); g_strfreev(alphas_str);

    // Run the pipeline
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    g_print("Running...\n");

    GMainLoop *loop = g_main_loop_new(NULL, FALSE);
    GstBus *bus = gst_element_get_bus(pipeline);
    gst_bus_add_watch(bus, bus_callback, loop);
    gst_object_unref(bus);

    if (!changes.empty()) g_timeout_add(100, update_alpha, NULL);

    g_main_loop_run(loop);

    // Cleanup
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    if (mozza) gst_object_unref(mozza);
    g_main_loop_unref(loop);
    return 0;
}