import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RecommendationSystem:
    def __init__(self, n_users=100, n_items=50):
        self.n_users = n_users
        self.n_items = n_items
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None

    def generate_ratings_data(self):
        """Generate synthetic user-item ratings"""
        print("\n📊 Generating synthetic ratings data...")

        np.random.seed(42)

        # Create sparse rating matrix (not all users rate all items)
        ratings_data = []

        for user_id in range(self.n_users):
            # Each user rates 20-40 items
            n_ratings = np.random.randint(20, 40)
            items_rated = np.random.choice(self.n_items, n_ratings, replace=False)

            for item_id in items_rated:
                # Ratings between 1-5
                # Create some pattern: users with similar IDs tend to like similar items
                base_rating = 3 + np.sin(user_id / 10 + item_id / 5)
                rating = np.clip(base_rating + np.random.normal(0, 0.5), 1, 5)

                ratings_data.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': round(rating, 1)
                })

        df = pd.DataFrame(ratings_data)
        print(f"✓ Generated {len(df)} ratings from {self.n_users} users on {self.n_items} items")
        print(f"✓ Sparsity: {1 - len(df) / (self.n_users * self.n_items):.2%}")

        return df

    def create_user_item_matrix(self, df):
        """Create user-item rating matrix"""
        self.user_item_matrix = df.pivot(
            index='user_id',
            columns='item_id',
            values='rating'
        ).fillna(0)

        return self.user_item_matrix

    def calculate_similarities(self):
        """Calculate user and item similarities"""
        print("\n🔗 Calculating similarity matrices...")

        # User similarity
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )

        # Item similarity
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )

        print("✓ User similarity matrix computed")
        print("✓ Item similarity matrix computed")

    def user_based_recommendations(self, user_id, n_recommendations=5):
        """Generate recommendations using user-based collaborative filtering"""
        # Get similar users
        similar_users = self.user_similarity[user_id].sort_values(ascending=False)[1:11]

        # Get items the user hasn't rated
        user_ratings = self.user_item_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index

        # Calculate weighted ratings for unrated items
        recommendations = {}
        for item in unrated_items:
            weighted_sum = 0
            similarity_sum = 0

            for similar_user, similarity in similar_users.items():
                rating = self.user_item_matrix.loc[similar_user, item]
                if rating > 0:
                    weighted_sum += similarity * rating
                    similarity_sum += similarity

            if similarity_sum > 0:
                recommendations[item] = weighted_sum / similarity_sum

        # Sort and return top N
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        return top_recommendations

    def item_based_recommendations(self, user_id, n_recommendations=5):
        """Generate recommendations using item-based collaborative filtering"""
        # Get items the user has rated
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_items = user_ratings[user_ratings > 0]

        # Get items the user hasn't rated
        unrated_items = user_ratings[user_ratings == 0].index

        # Calculate recommendations
        recommendations = {}
        for item in unrated_items:
            weighted_sum = 0
            similarity_sum = 0

            for rated_item, rating in rated_items.items():
                similarity = self.item_similarity.loc[item, rated_item]
                weighted_sum += similarity * rating
                similarity_sum += abs(similarity)

            if similarity_sum > 0:
                recommendations[item] = weighted_sum / similarity_sum

        # Sort and return top N
        top_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
        return top_recommendations

    def matrix_factorization_svd(self, n_factors=10):
        """Perform matrix factorization using SVD"""
        print(f"\n🔬 Performing SVD with {n_factors} latent factors...")

        # Normalize by subtracting mean
        user_ratings_mean = np.mean(self.user_item_matrix.values, axis=1)
        matrix_normalized = self.user_item_matrix.values - user_ratings_mean.reshape(-1, 1)

        # SVD
        U, sigma, Vt = svds(matrix_normalized, k=n_factors)
        sigma = np.diag(sigma)

        # Reconstruct matrix
        predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

        self.predicted_matrix = pd.DataFrame(
            predicted_ratings,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns
        )

        print("✓ SVD completed")
        return self.predicted_matrix

    def svd_recommendations(self, user_id, n_recommendations=5):
        """Generate recommendations using SVD"""
        # Get user's predicted ratings
        user_predictions = self.predicted_matrix.loc[user_id]

        # Get items user hasn't rated
        user_actual = self.user_item_matrix.loc[user_id]
        unrated_items = user_actual[user_actual == 0].index

        # Get top predictions for unrated items
        recommendations = user_predictions[unrated_items].sort_values(ascending=False)[:n_recommendations]

        return list(recommendations.items())

    def evaluate_recommendations(self, test_df):
        """Calculate RMSE on test set"""
        errors = []

        for _, row in test_df.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            actual_rating = row['rating']

            if user_id in self.predicted_matrix.index and item_id in self.predicted_matrix.columns:
                predicted_rating = self.predicted_matrix.loc[user_id, item_id]
                errors.append((actual_rating - predicted_rating) ** 2)

        rmse = np.sqrt(np.mean(errors))
        return rmse

    def visualize_similarity_matrices(self):
        """Visualize similarity matrices"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # User similarity (sample)
        sample_users = np.random.choice(self.user_similarity.index, 20, replace=False)
        user_sim_sample = self.user_similarity.loc[sample_users, sample_users]

        sns.heatmap(user_sim_sample, cmap='coolwarm', center=0, ax=axes[0], 
                   cbar_kws={'label': 'Similarity'})
        axes[0].set_title('User Similarity Matrix (Sample)', fontsize=12, fontweight='bold')

        # Item similarity (sample)
        sample_items = np.random.choice(self.item_similarity.index, 20, replace=False)
        item_sim_sample = self.item_similarity.loc[sample_items, sample_items]

        sns.heatmap(item_sim_sample, cmap='coolwarm', center=0, ax=axes[1],
                   cbar_kws={'label': 'Similarity'})
        axes[1].set_title('Item Similarity Matrix (Sample)', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig('similarity_matrices.png', dpi=300, bbox_inches='tight')
        print("\n✓ Similarity matrices visualization saved")
        plt.show()

    def run_pipeline(self):
        """Execute recommendation pipeline"""
        print("\n" + "="*60)
        print("Recommendation System Pipeline")
        print("="*60)

        # Generate data
        ratings_df = self.generate_ratings_data()

        # Split train/test
        train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)

        # Create matrix
        self.create_user_item_matrix(train_df)

        # Calculate similarities
        self.calculate_similarities()

        # Matrix factorization
        self.matrix_factorization_svd(n_factors=10)

        # Evaluate
        print("\n📊 Evaluating model...")
        rmse = self.evaluate_recommendations(test_df)
        print(f"✓ RMSE on test set: {rmse:.3f}")

        # Generate sample recommendations
        sample_user = 5
        print(f"\n{'='*60}")
        print(f"Sample Recommendations for User {sample_user}")
        print(f"{'='*60}")

        print(f"\nUser-Based CF:")
        user_recs = self.user_based_recommendations(sample_user, 5)
        for item, score in user_recs:
            print(f"  Item {item}: {score:.2f}")

        print(f"\nItem-Based CF:")
        item_recs = self.item_based_recommendations(sample_user, 5)
        for item, score in item_recs:
            print(f"  Item {item}: {score:.2f}")

        print(f"\nSVD-Based:")
        svd_recs = self.svd_recommendations(sample_user, 5)
        for item, score in svd_recs:
            print(f"  Item {item}: {score:.2f}")

        # Save results
        ratings_df.to_csv('ratings_data.csv', index=False)
        print(f"\n✓ Ratings data saved to 'ratings_data.csv'")

        # Visualize
        print("\n📈 Generating visualizations...")
        self.visualize_similarity_matrices()

        print("\n" + "="*60)
        print("Pipeline Complete!")
        print("="*60 + "\n")

if __name__ == "__main__":
    rec_system = RecommendationSystem(n_users=100, n_items=50)
    rec_system.run_pipeline()
