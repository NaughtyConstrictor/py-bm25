#ifndef BM25__H
#define BM25__H

#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>


namespace BM25 {

    class BM25_ {
    public:
        BM25_(const std::vector< std::vector<std::string> > &, double, double, bool);
        virtual ~BM25_() = default;
        int get_corpus_size() const;
        double get_avgdl() const;
        bool get_invert_index() const;
        double get_k1() const;
        double get_b() const;
        void set_k1(double);
        void set_b(double);
        std::unordered_map<std::string, double>& get_idfs();
        std::vector<int>& get_doc_len();
        std::vector< std::unordered_map<std::string, int> >& get_index();
        std::unordered_map<std::string, std::vector<int>> & get_inverted_index();
        Eigen::ArrayXd get_scores(std::vector<std::string> &);
        Eigen::ArrayXXd get_scores_batch(std::vector< std::vector<std::string> > &);
        virtual Eigen::ArrayXd get_scores_from_index(std::vector<std::string> &) = 0;
        virtual Eigen::ArrayXd get_scores_from_inverted_index(std::vector<std::string> &) = 0;
    
    protected:
        void build_index(const std::vector< std::vector<std::string> > &);
        void build_inverted_index(const std::vector < std::vector<std::string> > &);
        virtual void calc_idf(const std::unordered_map<std::string, int> &) = 0;
        int corpus_size;
        double avgdl;
        bool invert_index;
        double k1;
        double b;
        Eigen::ArrayXd cst;
        std::vector< std::unordered_map<std::string, int> > index;
        std::unordered_map< std::string, std::vector<int> > inverted_index;
        std::unordered_map<std::string, double> idfs;
        std::vector<int> doc_len;
        std::unordered_map<std::string, int> nd;
    };

    class BM25Okapi_ : public BM25_{
    public:
        BM25Okapi_(const std::vector< std::vector<std::string> > &, double, double, double, bool);
        double get_epsilon() const;
        void set_epsilon(double);

    private:
        double epsilon;
        void calc_idf(const std::unordered_map<std::string, int> &) override;
        Eigen::ArrayXd get_scores_from_index(std::vector<std::string> & query) override;
        Eigen::ArrayXd get_scores_from_inverted_index(std::vector<std::string> & query) override;
    };

    class BM25L_ : public BM25_{
    public:
        BM25L_(const std::vector< std::vector<std::string> > &, double, double, double, bool);
        double get_delta() const;
        void set_delta(double);

    private:
        double delta;
        void calc_idf(const std::unordered_map<std::string, int> &) override;
        Eigen::ArrayXd get_scores_from_index(std::vector<std::string> &) override;
        Eigen::ArrayXd get_scores_from_inverted_index(std::vector<std::string> &) override;
    };

    class BM25Plus_ : public BM25_{
    public:
        BM25Plus_(const std::vector< std::vector<std::string> > &, double, double, double, bool);
        double get_delta() const;
        void set_delta(double);

    private:
        double delta;
        void calc_idf(const std::unordered_map<std::string, int> &) override;
        Eigen::ArrayXd get_scores_from_index(std::vector<std::string> &) override;
        Eigen::ArrayXd get_scores_from_inverted_index(std::vector<std::string> &) override;
    };
}

#endif