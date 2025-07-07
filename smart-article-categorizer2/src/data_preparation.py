import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

class DataPreprocessor:
    def __init__(self) -> None:
        self.categories = ["Tech", 'Finance', 'Healthcare', 'Sports', 'Politics', 'Entertainment']
    
    def load_news_dataset(self):
        dataset = load_dataset("ag_news")
        return self.create_sample_dataset()

    def create_sample_dataset(self):
        """Create comprehensive sample dataset for demonstration"""
        sample_data = {
            'text': [
                # Tech articles (20 examples)
                "Apple releases new iPhone with advanced AI features and improved camera system for professional photography",
                "Google announces breakthrough in quantum computing research with 1000-qubit processor",
                "Microsoft launches new cloud computing platform for enterprise with enhanced security features",
                "Tesla unveils next-generation electric vehicle with autonomous driving capabilities",
                "Facebook introduces new virtual reality headset for gaming and social interaction",
                "Amazon develops new AI-powered recommendation system for personalized shopping",
                "Intel announces new processor architecture with significant performance improvements",
                "Samsung launches foldable smartphone with innovative display technology",
                "NVIDIA releases new graphics card optimized for artificial intelligence workloads",
                "SpaceX successfully launches satellite constellation for global internet coverage",
                "Oracle introduces new database management system with blockchain integration",
                "Adobe releases creative software suite with AI-powered design tools",
                "Cisco announces new networking equipment for 5G infrastructure deployment",
                "IBM develops quantum computer accessible through cloud platform",
                "Netflix implements machine learning algorithms for content recommendation",
                "Uber launches autonomous vehicle testing program in major cities",
                "Zoom introduces new video conferencing features with AI background removal",
                "Salesforce announces CRM platform with predictive analytics capabilities",
                "Palantir develops data analysis software for government intelligence agencies",
                "Twitter implements new content moderation system using natural language processing",
                
                # Finance articles (20 examples)
                "Stock market shows volatility amid inflation concerns and Federal Reserve policy changes",
                "Bitcoin reaches new all-time high as institutional adoption increases significantly",
                "Goldman Sachs reports record quarterly profits despite market uncertainty and volatility",
                "Federal Reserve announces interest rate adjustments to combat rising inflation",
                "Wall Street banks prepare for new regulatory requirements and compliance standards",
                "Cryptocurrency market experiences significant growth in institutional investment",
                "JP Morgan Chase reports strong earnings despite economic headwinds",
                "Visa and Mastercard announce new digital payment solutions for merchants",
                "BlackRock launches new ESG-focused investment fund for sustainable finance",
                "Morgan Stanley acquires fintech startup to expand digital banking services",
                "Citigroup implements new risk management system for credit assessment",
                "Wells Fargo announces restructuring plan to improve operational efficiency",
                "American Express introduces new rewards program for premium cardholders",
                "PayPal expands cryptocurrency services to include additional digital assets",
                "Charles Schwab reports increased trading volume during market volatility",
                "Fidelity Investments launches new robo-advisor platform for retail investors",
                "Bank of America announces digital transformation initiative for customer service",
                "Vanguard introduces new index funds with lower expense ratios",
                "TD Ameritrade reports record account openings during market rally",
                "E*TRADE launches new options trading platform with advanced analytics",
                
                # Healthcare articles (20 examples)
                "New cancer treatment shows promising results in clinical trials with improved survival rates",
                "FDA approves breakthrough drug for Alzheimer's disease treatment after extensive testing",
                "Medical researchers discover novel approach to diabetes management using gene therapy",
                "Hospital implements new telemedicine platform for remote patient care and consultation",
                "Pharmaceutical company develops vaccine for emerging infectious disease outbreak",
                "Healthcare AI system improves diagnostic accuracy in radiology and medical imaging",
                "Johnson & Johnson announces new drug development for autoimmune disorders",
                "Pfizer reports positive results from COVID-19 vaccine booster study",
                "Merck develops new immunotherapy treatment for advanced cancer patients",
                "Roche introduces diagnostic test for early detection of genetic disorders",
                "Novartis launches gene therapy program for rare disease treatment",
                "AstraZeneca announces partnership for personalized medicine development",
                "Gilead Sciences reports breakthrough in HIV treatment research",
                "Amgen develops new biologic drug for inflammatory conditions",
                "Bristol-Myers Squibb introduces immunotherapy combination therapy",
                "Moderna announces mRNA technology platform for vaccine development",
                "Regeneron develops monoclonal antibody treatment for infectious diseases",
                "Biogen reports progress in multiple sclerosis treatment research",
                "Vertex Pharmaceuticals launches cystic fibrosis treatment program",
                "Illumina introduces next-generation sequencing for genetic testing",
                
                # Sports articles (20 examples)
                "Basketball championship finals draw record viewers and social media engagement worldwide",
                "Soccer team wins championship with dramatic overtime victory in penalty shootout",
                "Olympic athletes break world records in swimming competition with new techniques",
                "Baseball player signs record-breaking contract with major league team for 10 years",
                "Tennis tournament features intense rivalry between top-ranked players in final match",
                "Football team advances to playoffs with last-minute touchdown in overtime period",
                "Golf tournament sees unexpected victory by rookie player with impressive performance",
                "Hockey team clinches division title with overtime goal in final regular season game",
                "Olympic gymnastics team wins gold medal with perfect routine execution",
                "Soccer league announces new broadcasting rights deal worth billions of dollars",
                "Basketball star returns from injury to lead team to championship victory",
                "Baseball pitcher throws perfect game for first time in franchise history",
                "Tennis player wins Grand Slam tournament with dominant performance throughout",
                "Football quarterback sets new passing record in single season performance",
                "Olympic swimming relay team breaks world record with synchronized technique",
                "Basketball coach leads underdog team to unexpected playoff appearance",
                "Soccer player scores hat trick in championship match with spectacular goals",
                "Baseball team wins World Series with dramatic comeback in final game",
                "Tennis tournament attracts record attendance with star-studded player field",
                "Olympic track and field athlete wins multiple gold medals in different events",
                
                # Politics articles (20 examples)
                "Senate passes new infrastructure bill with bipartisan support and funding allocation",
                "President announces new foreign policy initiatives for trade relations with allies",
                "Congress debates healthcare reform legislation in heated session with amendments",
                "Supreme Court issues landmark ruling on constitutional rights and civil liberties",
                "Local government implements new environmental protection policies for sustainability",
                "International summit addresses climate change and economic cooperation agreements",
                "House of Representatives votes on budget bill with party-line division",
                "State legislature passes education reform bill with teacher union support",
                "Mayor announces urban development plan for affordable housing construction",
                "Governor signs executive order for renewable energy transition in state",
                "Federal agency implements new regulations for financial industry oversight",
                "Diplomatic delegation meets with foreign leaders for trade agreement negotiations",
                "Political party announces platform changes for upcoming election campaign",
                "Judicial committee reviews Supreme Court nominee qualifications and background",
                "Municipal council approves zoning changes for commercial development project",
                "State attorney general files lawsuit against pharmaceutical companies",
                "Congressional committee investigates corporate influence on policy decisions",
                "Presidential candidate releases detailed policy proposals for economic reform",
                "International organization addresses human rights violations in conflict zones",
                "Local election board implements new voting system for improved security",
                
                # Entertainment articles (20 examples)
                "Hollywood blockbuster breaks box office records worldwide with massive opening weekend",
                "Streaming service releases highly anticipated original series with star-studded cast",
                "Music festival attracts thousands of fans with star-studded lineup and performances",
                "Award show celebrates achievements in film and television industry with glamorous ceremony",
                "Video game developer announces sequel to popular franchise with enhanced graphics",
                "Celebrity couple announces engagement on social media platform with romantic proposal",
                "Broadway musical wins Tony Award for best new production with innovative staging",
                "Comedy special receives critical acclaim for groundbreaking humor and social commentary",
                "Reality television show becomes viral sensation with unexpected plot twists",
                "Animated film receives Oscar nomination for best picture with universal appeal",
                "Rock band announces world tour with stadium venues and elaborate stage production",
                "Drama series finale draws record viewership with shocking plot revelations",
                "Comedy film becomes sleeper hit with word-of-mouth marketing and audience support",
                "Pop star releases new album with chart-topping singles and innovative sound",
                "Action movie franchise continues with successful sequel and special effects",
                "Talk show host interviews controversial figure with ratings spike and media attention",
                "Documentary filmmaker wins award for investigative journalism and storytelling",
                "Stand-up comedian performs sold-out shows with observational humor and audience interaction",
                "Drama actor receives Oscar nomination for transformative performance and character study",
                "Music producer collaborates with multiple artists for chart-topping compilation album"
            ],
            'category': [
                'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech',
                'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech', 'Tech',
                'Finance', 'Finance', 'Finance', 'Finance', 'Finance', 'Finance', 'Finance', 'Finance', 'Finance', 'Finance',
                'Finance', 'Finance', 'Finance', 'Finance', 'Finance', 'Finance', 'Finance', 'Finance', 'Finance', 'Finance',
                'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare',
                'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare', 'Healthcare',
                'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports',
                'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports', 'Sports',
                'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics',
                'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics', 'Politics',
                'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment',
                'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment', 'Entertainment'
            ]
        }
        return pd.DataFrame(sample_data)

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespaces
        text = ' '.join(text.split())
        return text
    
    def prepare_data(self):
        """Main data preparation pipeline"""
        df = self.load_news_dataset()
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['category'], 
            test_size=0.2, random_state=42, stratify=df['category']
        )
        
        return X_train, X_test, y_train, y_test   