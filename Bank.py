import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# st.set_page_config(
#     page_title="Bank Marketing Success Analysis",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     page_icon="📊",
# )

def bank_page():

    st.title("Bank Marketing Success Analysis")

    # sidebar
    st.sidebar.header("Bank Marketing Success Analysis")
    st.sidebar.image("gadgets/data_analysis.jpeg")
    st.sidebar.markdown(
        "Made By : Basme Zantout, Zeynep Sude Bal, Gizem Yüksel, Ahmet Tokgöz"
    )

    # Load data
    df = pd.read_csv(r'datasets\bank-additional.csv', sep=';')


    def is_contacted_before(x):
        if x == 999:
            return "no"
        else:
            return "yes"


    df["contacted_before"] = df["pdays"].apply(is_contacted_before)
    df.drop("pdays", axis=1, inplace=True)

    # Body

    # Matrix
    st.subheader("Important Metrics")

    col1, col2, col3, col4 = st.columns(4)  # For Horizontal matrix
    col1.metric(
        "Current Campain Successful Deposits ",
        np.round(df[df["y"] == "yes"]["y"].count()),
    )
    col2.metric(
        "Current Campain Successful Deposits %",
        np.round(df[df["y"] == "yes"]["y"].count() / len(df) * 100, 2),
    )
    col3.metric(
        "Previous Campain Successful Deposits",
        np.round(df[df["poutcome"] == "success"]["poutcome"].count()),
    )
    col4.metric(
        "Previous Campain Successful Deposits %",
        np.round(df[df["poutcome"] == "success"]["poutcome"].count() / len(df) * 100, 2),
    )


    # Function to plot histogram
    def plot_histogram(df, column):
        fig = px.histogram(
            df,
            x=column,
            nbins=20,
            title=f"Histogram of {column}",
            labels={column: column},
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)


    # row 2
    c1, c2 = st.columns(2)
    with c1:
        plot_histogram(df, "age")
    with c2:
        plot_histogram(df, "campaign")

    # row 3
    c1, c2 = st.columns(2)
    with c1:
        plot_histogram(df, "nr.employed")
    with c2:
        plot_histogram(df, "previous")

    # row 4
    c1, c2 = st.columns(2)
    with c1:
        plot_histogram(df, "emp.var.rate")
    with c2:
        plot_histogram(df, "cons.price.idx")

    # row 5
    c1, c2 = st.columns(2)
    with c1:
        plot_histogram(df, "cons.conf.idx")
    with c2:
        plot_histogram(df, "euribor3m")


    # row 6
    plot_histogram(df, "duration")


    # Categorical Columns


    # Function to plot a Pie chart for a categorical column
    def plot_pie_chart(df, column):
        # Count the occurrences of each category in the column
        fig = px.pie(
            df,
            names=column,
            title=f"Pie Chart of {column}",
            hole=0.3,  # Optional: to create a donut chart
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)


    # Function to plot a grouped histogram of a categorical column against a target variable
    def plot_categorical_vs_target_histogram(df, categorical_column, target_column):
        # Creating the grouped histogram
        fig = px.histogram(
            df,
            x=categorical_column,
            color=target_column,
            barmode="group",
            title=f"{categorical_column} vs Deposit Success (Yes/No)",
            text_auto=True,
            template="plotly_white",
        )
        # Displaying the figure in Streamlit
        st.plotly_chart(fig, use_container_width=True)


    # row 7
    c1, c2 = st.columns(2)
    with c1:
        plot_pie_chart(df, "marital")
    with c2:
        plot_pie_chart(df, "contacted_before")

    plot_categorical_vs_target_histogram(df, "job", "y")


    # row 8
    c1, c2 = st.columns(2)
    with c1:
        plot_pie_chart(df, "housing")
    with c2:
        plot_pie_chart(df, "default")

    plot_categorical_vs_target_histogram(df, "education", "y")

    # row 9
    c1, c2 = st.columns(2)
    with c1:
        plot_pie_chart(df, "loan")
    with c2:
        plot_pie_chart(df, "contact")

    plot_categorical_vs_target_histogram(df, "day_of_week", "y")


    # row 10
    c1, c2 = st.columns(2)
    with c1:
        plot_pie_chart(df, "poutcome")
    with c2:
        plot_pie_chart(df, "y")

    plot_categorical_vs_target_histogram(df, "month", "y")

    # Insights

    st.header("Insights")


    insights = """

    1. **Job Role and Subscription Rate**:
        - Students and retired individuals are more likely to subscribe to a term deposit.

    2. **Marital Status Influence**:
        - Single clients have a higher subscription rate compared to married or divorced clients.

    3. **Education Level Impact**:
        - Higher education levels, such as high school, university, and personal courses, have higher subscription rates compared to primary education.

    4. **Housing Loan and Personal Loan Factors**:
        - Clients with existing housing loans are more likely to subscribe to term deposits.
        - Clients without existing personal loans are more likely to subscribe to term deposits.

    5. **Contact Method Effectiveness**:
        - Contacting clients via cell phones yields higher subscription rates.

    6. **Time-Related Factors**:
        - **Month**: Subscription rates are higher during September, October, December, and March.

    7. **Effect of Previous Campaign Outcomes**:
        - Customers who had a successful interaction in the previous campaign are more likely to subscribe to the current one.

    8. **Economic Indicators' Influence**:
        - Economic context variables, such as the employment variation rate (`emp.var.rate`), consumer price index (`cons.price.idx`), number of employees (`nr.employed`), and Euribor 3-month rate (`euribor3m`), have a negative impact on the subscription rate.
        - On the other hand, the consumer confidence index (`cons.conf.idx`) has a positive impact on the subscription rate.

    9. **Duration of Last Contact**:
        - The duration of the last contact is a strong indicator of the outcome. Longer conversation durations have a higher chance of subscription.

    10. **Campaign Performance Metrics**:
        - More calls during the current campaign are more likely to result in an unsuccessful deposit.
        - On the other hand, more calls before the current campaign are more likely to result in a successful deposit.
    """

    st.markdown(insights)
