import streamlit as st
import plotly.graph_objs as go
import scipy.signal as scipy

import lps_rvt.dataloader as rvt_loader
import lps_rvt.types as rvt

class Homepage:
    def __init__(self):
        st.set_page_config(page_title="RVT", layout="wide")
        self.loader = rvt_loader.DataLoader()

    def show_dataloader_selection(self):
        """Creates a selection interface in Streamlit's sidebar and displays filtered files"""
        with st.expander("Configuração do Teste", expanded=False):
            ammunition_options = [e.name for e in rvt.Ammunition]
            subset_options = [e.name for e in rvt.Subset]

            selected_ammunition = st.multiselect("Select Ammunition Types",
                                                 ammunition_options,
                                                 default=[rvt.Ammunition.EXSUP.name])
            selected_buoys = st.multiselect("Select Buoy IDs", options=range(1, 6))
            selected_subsets = st.multiselect("Select Subsets",
                                              subset_options,
                                              default=[rvt.Subset.TRAIN.name])

            file_types = [rvt.Ammunition[t] for t in selected_ammunition] \
                                if selected_ammunition else None
            buoys = selected_buoys if selected_buoys else None
            subsets = [rvt.Subset[s] for s in selected_subsets] \
                                if selected_subsets else None

        return self.loader.get_files(file_types, buoys, subsets)

    def plot_data(self, file_id):
        """Generates a Plotly plot with the data"""
        fs, data = self.loader.get_data(file_id)

        num_samples = 400000
        data_resampled = scipy.resample(data, num_samples)

        original_samples = len(data)
        resampling_factor = original_samples / num_samples
        new_fs = fs / resampling_factor

        time_axis = [i / new_fs for i in range(len(data_resampled))]

        trace_signal = go.Scatter(x=time_axis, y=data_resampled, mode='lines', name='Signal Data')

        expected_detections, _ = self.loader.get_critical_points(file_id, new_fs)

        shapes = []
        for d in expected_detections:
            shapes.append(
                dict(
                    type="line",
                    x0=time_axis[int(d)],
                    y0=min(data_resampled),
                    x1=time_axis[int(d)],
                    y1=max(data_resampled),
                    line=dict(color="green", width=2, dash="dot")
                )
            )

        layout = go.Layout(
            title=f"Arquivo {file_id}",
            xaxis=dict(title="Time (seconds)"),
            yaxis=dict(title="Amplitude"),
            showlegend=True,
            shapes=shapes
        )

        fig = go.Figure(data=[trace_signal], layout=layout)
        st.plotly_chart(fig)

    def run(self):
        show_results = False
        with st.sidebar:
            selected_files = self.show_dataloader_selection()

            with st.expander("Configuração do pre-processamento", expanded=True):
                st.write("...")

            with st.expander("Configuração do detector", expanded=True):
                st.write("...")

            if st.button("Play"):
                st.write("### Selected Files for Processing")
                st.write(selected_files)
                show_results = True

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("<h1 style='text-align: center;'>RVT</h1>", unsafe_allow_html=True)
        with col2:
            st.image("./data/logo.png", width=300)

        st.title("Main Content Area")
        st.write("Welcome to the Streamlit app with an expandable sidebar menu!")

        if show_results:
            for file_id in selected_files:
                self.plot_data(file_id)

if __name__ == "__main__":
    app = Homepage()
    app.run()
