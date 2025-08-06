import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visuals
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SuperstoreVisualizer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare data for analysis"""
        # Convert dates
        self.df['Order Date'] = pd.to_datetime(self.df['Order Date'])
        self.df['Ship Date'] = pd.to_datetime(self.df['Ship Date'])
        
        # Create additional columns
        self.df['Year'] = self.df['Order Date'].dt.year
        self.df['Month'] = self.df['Order Date'].dt.month
        self.df['Quarter'] = self.df['Order Date'].dt.quarter
        self.df['Profit Margin'] = (self.df['Profit'] / self.df['Sales']) * 100
        
        print("Data prepared successfully!")
        print(f"Dataset shape: {self.df.shape}")
        
    def create_executive_summary(self):
        """Create executive summary with key metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Superstore Business Performance - Executive Summary', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Sales by Category (Pie Chart)
        category_sales = self.df.groupby('Category')['Sales'].sum()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        wedges, texts, autotexts = ax1.pie(category_sales.values, labels=category_sales.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Sales Distribution by Category', fontsize=14, fontweight='bold', pad=20)
        
        # 2. Monthly Sales Trend
        monthly_sales = self.df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
        monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))
        ax2.plot(monthly_sales['Date'], monthly_sales['Sales'], marker='o', linewidth=3, markersize=6, color='#FF6B6B')
        ax2.set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Sales ($)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Regional Performance
        regional_data = self.df.groupby('Region').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
        x = np.arange(len(regional_data))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, regional_data['Sales'], width, label='Sales', color='#4ECDC4', alpha=0.8)
        ax3_twin = ax3.twinx()
        bars2 = ax3_twin.bar(x + width/2, regional_data['Profit'], width, label='Profit', color='#FF6B6B', alpha=0.8)
        
        ax3.set_title('Sales vs Profit by Region', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Region')
        ax3.set_ylabel('Sales ($)', color='#4ECDC4')
        ax3_twin.set_ylabel('Profit ($)', color='#FF6B6B')
        ax3.set_xticks(x)
        ax3.set_xticklabels(regional_data['Region'])
        
        # 4. Top 10 States by Profit
        state_profit = self.df.groupby('State')['Profit'].sum().sort_values(ascending=True).tail(10)
        bars = ax4.barh(range(len(state_profit)), state_profit.values, color='#45B7D1', alpha=0.8)
        ax4.set_yticks(range(len(state_profit)))
        ax4.set_yticklabels(state_profit.index)
        ax4.set_title('Top 10 States by Profit', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Profit ($)')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width + 50, bar.get_y() + bar.get_height()/2, f'${width:,.0f}', 
                    ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('executive_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_profitability_analysis(self):
        """Deep dive into profitability"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Profitability Deep Dive Analysis', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Profit Margin by Category and Segment
        profit_data = self.df.groupby(['Category', 'Segment'])['Profit Margin'].mean().unstack()
        profit_data.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Average Profit Margin by Category & Segment', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Category')
        ax1.set_ylabel('Profit Margin (%)')
        ax1.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Sales vs Profit Scatter
        scatter = ax2.scatter(self.df['Sales'], self.df['Profit'], 
                            c=self.df['Category'].astype('category').cat.codes, 
                            alpha=0.6, s=50)
        ax2.set_title('Sales vs Profit Relationship', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Sales ($)')
        ax2.set_ylabel('Profit ($)')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.df['Sales'], self.df['Profit'], 1)
        p = np.poly1d(z)
        ax2.plot(self.df['Sales'], p(self.df['Sales']), "r--", alpha=0.8, linewidth=2)
        
        # 3. Discount Impact on Profit
        discount_profit = self.df.groupby('Discount').agg({'Profit': 'mean', 'Sales': 'count'}).reset_index()
        bars = ax3.bar(discount_profit['Discount'], discount_profit['Profit'], 
                      color='#FF6B6B', alpha=0.7, width=0.02)
        ax3.set_title('Impact of Discount on Average Profit', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Discount Rate')
        ax3.set_ylabel('Average Profit ($)')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'${height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Quarterly Profit Trend
        quarterly_profit = self.df.groupby(['Year', 'Quarter'])['Profit'].sum().reset_index()
        quarterly_profit['Period'] = quarterly_profit['Year'].astype(str) + '-Q' + quarterly_profit['Quarter'].astype(str)
        
        ax4.plot(quarterly_profit['Period'], quarterly_profit['Profit'], 
                marker='o', linewidth=3, markersize=8, color='#4ECDC4')
        ax4.set_title('Quarterly Profit Trend', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Quarter')
        ax4.set_ylabel('Profit ($)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('profitability_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_customer_insights(self):
        """Customer segment analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Segment Insights', fontsize=20, fontweight='bold', y=0.98)
        
        # 1. Segment Performance
        segment_data = self.df.groupby('Segment').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Order ID': 'nunique'
        }).reset_index()
        
        x = np.arange(len(segment_data))
        width = 0.25
        
        ax1.bar(x - width, segment_data['Sales']/1000, width, label='Sales (K$)', color='#4ECDC4', alpha=0.8)
        ax1.bar(x, segment_data['Profit']/1000, width, label='Profit (K$)', color='#FF6B6B', alpha=0.8)
        ax1.bar(x + width, segment_data['Order ID'], width, label='Orders', color='#45B7D1', alpha=0.8)
        
        ax1.set_title('Performance by Customer Segment', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Segment')
        ax1.set_ylabel('Value')
        ax1.set_xticks(x)
        ax1.set_xticklabels(segment_data['Segment'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Ship Mode Preferences
        ship_mode_data = self.df.groupby(['Segment', 'Ship Mode']).size().unstack(fill_value=0)
        ship_mode_data.plot(kind='bar', stacked=True, ax=ax2, width=0.8)
        ax2.set_title('Shipping Preferences by Segment', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Segment')
        ax2.set_ylabel('Number of Orders')
        ax2.legend(title='Ship Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Average Order Value by Segment
        aov_data = self.df.groupby('Segment')['Sales'].mean().sort_values(ascending=True)
        bars = ax3.barh(range(len(aov_data)), aov_data.values, color='#45B7D1', alpha=0.8)
        ax3.set_yticks(range(len(aov_data)))
        ax3.set_yticklabels(aov_data.index)
        ax3.set_title('Average Order Value by Segment', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Average Order Value ($)')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax3.text(width + 5, bar.get_y() + bar.get_height()/2, f'${width:.0f}', 
                    ha='left', va='center', fontweight='bold')
        
        # 4. Category Preferences by Segment
        category_segment = pd.crosstab(self.df['Segment'], self.df['Category'], normalize='index') * 100
        category_segment.plot(kind='bar', ax=ax4, width=0.8)
        ax4.set_title('Category Preferences by Segment (%)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Segment')
        ax4.set_ylabel('Percentage of Orders')
        ax4.legend(title='Category')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('customer_insights.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sales by Region', 'Monthly Trend', 'Category Performance', 'Profit Margin Analysis'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # 1. Sales by Region
        regional_sales = self.df.groupby('Region')['Sales'].sum().reset_index()
        fig.add_trace(
            go.Bar(x=regional_sales['Region'], y=regional_sales['Sales'], 
                   name='Sales by Region', marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. Monthly Trend
        monthly_data = self.df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
        monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(day=1))
        fig.add_trace(
            go.Scatter(x=monthly_data['Date'], y=monthly_data['Sales'], 
                      mode='lines+markers', name='Monthly Sales', line=dict(color='red')),
            row=1, col=2
        )
        
        # 3. Category Performance
        category_data = self.df.groupby('Category').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
        fig.add_trace(
            go.Bar(x=category_data['Category'], y=category_data['Sales'], 
                   name='Sales', marker_color='lightgreen'),
            row=2, col=1
        )
        
        # 4. Profit Margin by Category
        for category in self.df['Category'].unique():
            category_data = self.df[self.df['Category'] == category]['Profit Margin']
            fig.add_trace(
                go.Box(y=category_data, name=category),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Superstore Interactive Dashboard",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        fig.write_html('interactive_dashboard.html')
        print("Interactive dashboard saved as 'interactive_dashboard.html'")
        
    def generate_insights_report(self):
        """Generate key business insights"""
        insights = []
        
        # Total metrics
        total_sales = self.df['Sales'].sum()
        total_profit = self.df['Profit'].sum()
        profit_margin = (total_profit / total_sales) * 100
        
        insights.append(f"EXECUTIVE SUMMARY")
        insights.append(f"• Total Sales: ${total_sales:,.2f}")
        insights.append(f"• Total Profit: ${total_profit:,.2f}")
        insights.append(f"• Overall Profit Margin: {profit_margin:.1f}%")
        insights.append("")
        
        # Best performing category
        category_performance = self.df.groupby('Category').agg({
            'Sales': 'sum', 'Profit': 'sum'
        }).reset_index()
        best_category = category_performance.loc[category_performance['Profit'].idxmax(), 'Category']
        
        insights.append(f"KEY FINDINGS")
        insights.append(f"• Best performing category: {best_category}")
        
        # Regional insights
        regional_profit = self.df.groupby('Region')['Profit'].sum()
        best_region = regional_profit.idxmax()
        worst_region = regional_profit.idxmin()
        
        insights.append(f"• Strongest region: {best_region} (${regional_profit[best_region]:,.0f} profit)")
        insights.append(f"• Opportunity region: {worst_region} (${regional_profit[worst_region]:,.0f} profit)")
        
        # Discount impact
        high_discount = self.df[self.df['Discount'] > 0.2]['Profit'].mean()
        no_discount = self.df[self.df['Discount'] == 0]['Profit'].mean()
        
        insights.append(f"• High discounts (>20%) reduce average profit by ${no_discount - high_discount:.0f}")
        
        # Customer segment insights
        segment_aov = self.df.groupby('Segment')['Sales'].mean()
        best_segment = segment_aov.idxmax()
        
        insights.append(f"• Highest value segment: {best_segment} (${segment_aov[best_segment]:.0f} avg order)")
        insights.append("")
        
        insights.append(f"RECOMMENDATIONS")
        insights.append(f"• Focus marketing efforts on {best_region} region expansion")
        insights.append(f"• Optimize discount strategy - high discounts hurt profitability")
        insights.append(f"• Develop {best_segment} segment with premium offerings")
        insights.append(f"• Investigate {worst_region} region challenges and opportunities")
        
        # Save insights to file
        with open('business_insights.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(insights))
        
        print("BUSINESS INSIGHTS GENERATED:")
        print("=" * 50)
        for insight in insights:
            print(insight)
        
    def create_all_visualizations(self):
        """Create all visualizations"""
        print("Creating Executive Summary...")
        self.create_executive_summary()
        
        print("Creating Profitability Analysis...")
        self.create_profitability_analysis()
        
        print("Creating Customer Insights...")
        self.create_customer_insights()
        
        print("Creating Interactive Dashboard...")
        self.create_interactive_dashboard()
        
        print("Generating Business Insights...")
        self.generate_insights_report()
        
        print("\n✅ All visualizations created successfully!")
        print("📁 Files generated:")
        print("  • executive_summary.png")
        print("  • profitability_analysis.png") 
        print("  • customer_insights.png")
        print("  • interactive_dashboard.html")
        print("  • business_insights.txt")

# Run the visualization suite
if __name__ == "__main__":
    visualizer = SuperstoreVisualizer('superstore_data.csv')
    visualizer.create_all_visualizations()