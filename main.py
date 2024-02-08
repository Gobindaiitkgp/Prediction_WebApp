import streamlit as st
from streamlit_option_menu import option_menu
import Machining_operation, Cutting_tool, Machine_tool, Machining_parameter

st.set_page_config(
    page_title="Process plan Generation",
)

class MultiApp:

    def __init__(self):
        self.predictions = {
            "Machining Operations": [],
            "Cutting Tools": [],
            "Machine Tools": [],
            "Machining Parameters": []
        }

    def run(self):
        with st.sidebar:        
            app = option_menu(
            menu_title='Process plan Generation',
            options=['Machining operation','Cutting tool','Machine tool','Machining parameter', 'Results'],
            default_index=1,
            styles={
                "container": {"padding": "5px !important", "background-color": "black"},
                "icon": {"color": "white", "font-size": "23px"}, 
                "nav-link": {"color": "white", "font-size": "20px", "text-align": "left", "margin": "0px", "--hover-color": "blue"},
                "nav-link-selected": {"background-color": "#02ab21"},
                "menu-title": {"color": "orange"}  # Adding custom style for the menu title
            }
        )


        if app == "Machining operation":
            self.predictions["Machining Operations"] = Machining_operation.app()
        elif app == "Cutting tool":
            self.predictions["Cutting Tools"] = Cutting_tool.app()    
        elif app == "Machine tool":
            self.predictions["Machine Tools"] = Machine_tool.app()        
        elif app == "Machining parameter":
            self.predictions["Machining Parameters"] = Machining_parameter.app()
        elif app == "Results":
            self.display_results()

    def display_results(self):
        st.title("Results Page")

        # Display the results in columns
        col1, col2, col3, col4 = st.columns(4)

        # Display Machining Operations
        col1.header("Machining Operations")
        if self.predictions["Machining Operations"]:
            col1.write(self.predictions["Machining Operations"])
        else:
            col1.write("No predictions available for Machining Operations")

        # Display Cutting Tools
        col2.header("Cutting Tools")
        if self.predictions["Cutting Tools"]:
            col2.write(self.predictions["Cutting Tools"])
        else:
            col2.write("No predictions available for Cutting Tools")

        # Display Machine Tools
        col3.header("Machine Tools")
        if self.predictions["Machine Tools"]:
            col3.write(self.predictions["Machine Tools"])
        else:
            col3.write("No predictions available for Machine Tools")

        # Display Machining Parameters
        col4.header("Machining Parameters")
        if self.predictions["Machining Parameters"]:
            col4.write(self.predictions["Machining Parameters"])
        else:
            col4.write("No predictions available for Machining Parameters")

# Run the multi-app
multi_app = MultiApp()
multi_app.run()
