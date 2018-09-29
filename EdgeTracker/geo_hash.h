// Geometric hashing example
// Obtained from http://www.cs.rpi.edu/academics/courses/spring08/cs2/homework/09/

#ifndef geo_hash_h_
#define geo_hash_h_

#include <list>
#include <vector>

using namespace std;

// A model-basis identifier
class model_basis {
public:
    model_basis(int in_model = -1, vector<int> in_basis = {}, vector<int> in_view = {}) : model(in_model), basis(in_basis), view(in_view) {}
    int model;
    vector<int> basis, view;
    
    bool operator == (const model_basis& mb) const
    {
        return (model == mb.model && basis == mb.basis && view == mb.view);
    }
};

//  A point location in 2d
class point {
public:
  point( float in_x, float in_y, int in_id = -1, model_basis in_model_basis = model_basis() ) : m_x(in_x), m_y(in_y), m_id(in_id), m_model_basis(in_model_basis) {}
  point() : m_x(0), m_y(0), m_id(-1) {}
  float x() const { return m_x; }
  float y() const { return m_y; }
  int getID() const { return m_id; }
  model_basis modelBasis() const { return m_model_basis; }
private:
  float m_x, m_y;
  int m_id;
  model_basis m_model_basis;
};


//  The index for a bin in a 2d grid
class bin_index {
public:
  bin_index( int in_i, int in_j ) : m_i(in_i), m_j(in_j) {}
  bin_index() : m_i(0), m_j(0) {}
  int i() const { return m_i; }
  int j() const { return m_j; }
  bool equals(bin_index bi) { return (i() == bi.i() && j() == bi.j()); }
private:
  int m_i, m_j;
};

class geo_hash {
public:
  //  Construct a geometric hash with square bins having the specified
  //  bin width.
  geo_hash( float bin_width=10.0, int numRotBins_in = 10 );

  //  Add a point to the geometric hash
  void add_point( point loc );

  //  Add a vector of points to the geometric hash
  void add_points( std::vector<point> const& locs );

  //  Find all points in the geometric hash that fall within the given
  //  circle.  Order them by increasing x and for ties, by increasing
  //  y
  std::vector<point> points_in_circle( point center, float radius ) const;

  //  Find all points in the geometric hash that fall within the given
  //  rectangle defined by the min_point (smallest x and y) and the
  //  max_point (greatest x and y).  Order the points found by
  //  increasing x and for ties, by increasing y
  std::vector<point> points_in_rectangle( point min_point, point max_point ) const;

  //  Erase the points that fall within the given circle
  int erase_points( point center, float radius=1e-6 );

  //  Find the bin index associated with a given point location
  bin_index point_to_bin( point loc ) const;

  //  Find the hash value of the given point location
  unsigned int hash_value( point loc ) const;

  //  What points are in the bin associated with the given point
  //  location.
  std::vector<point> points_in_bin( point loc ) const;

  //  How many non-empty bins are there?
  int num_non_empty() const;

  //  How many points are in the geometric hash?
  int num_points() const;

  //  What is the size of the hash table?
  int table_size() const;  

private:
  //  This is an internal record for an entry in the table.
  struct table_entry {
  public:
    bin_index bin;
    std::list<point> points;
  };

private:
  //  Iterator and cons iterator typedefs for the hash table
  typedef std::list<table_entry>::iterator table_entry_iterator;
  typedef std::list<table_entry>::const_iterator const_table_entry_iterator;

  //  Compute the hash value for the given bin index
  unsigned int hash_value( bin_index bin ) const;

  //  Find the table location and list iterator within the table for
  //  the given point.  Used when changes to the table are possible.
  bool find_entry_iterator( point                  loc, 
                            int                  & table_index,
                            table_entry_iterator & itr);

  //  Find the table location and list iterator within the table for
  //  the given point.  Used when changes to the table are not
  //  possible.
  bool find_entry_iterator( point                        loc, 
                            int                        & table_index,
                            const_table_entry_iterator & itr) const;

  //  Find the table location and list iterator within the table for
  //  the given bin.  Used when changes to the table are possible.
  bool find_entry_iterator( bin_index              bin, 
                            int                  & table_index,
                            table_entry_iterator & itr);

  //  Find the table location and list iterator within the table for
  //  the given bin.  Used when changes to the table are not
  //  possible.
  bool find_entry_iterator( bin_index                    bin,
                            int                        & table_index,
                            const_table_entry_iterator & itr) const;


private:
  //  The table itself
  std::vector< std::list<table_entry> > m_table;

  //  Size of the square bins
  float m_width;

  //  Counters
  int   m_num_bin_entries, m_num_points;
    
public:
  float width() { return m_width; };
  
  // Number of view bins (around the x-axis; /2 for y-axis)
  int numRotBins;
};


#endif
