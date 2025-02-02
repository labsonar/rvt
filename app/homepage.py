import streamlit as st

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

    def run(self):
        with st.sidebar:
            selected_files = self.show_dataloader_selection()

            with st.expander("Configuração do pre-processamento", expanded=True):
                st.write("...")

            with st.expander("Configuração do detector", expanded=True):
                st.write("...")

            if st.button("Play"):
                st.write("### Selected Files for Processing")
                st.write(selected_files)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown("<h1 style='text-align: center;'>RVT</h1>", unsafe_allow_html=True)
        with col2:
            st.image("./data/logo.png", width=300)

        st.title("Main Content Area")
        st.write("Welcome to the Streamlit app with an expandable sidebar menu!")

if __name__ == "__main__":
    app = Homepage()
    app.run()
