// Geometric hashing example
// Obtained from http://www.cs.rpi.edu/academics/courses/spring08/cs2/homework/09/

#include "geo_hash.h"

#include <algorithm>
#include <cmath>

bool operator< ( point const& left, point const& right )
{
  return left.x() < right.x() ||
    ( left.x() == right.x() && left.y() < right.y() );
}

bool operator== ( bin_index const& left, bin_index const& right )
{
  return left.i() == right.i() && left.j() == right.j();
}


geo_hash::geo_hash( float bin_width, int numRotBins_in )
  : m_table(4), m_width(bin_width), m_num_bin_entries(0), m_num_points(0), numRotBins(numRotBins_in)
{}


void
geo_hash::add_point( point loc )
{
  int            index;
  table_entry_iterator itr;

  //  Find the bin.
  if ( this->find_entry_iterator( loc, index, itr ) )
    {
      // If it is already there, just add the point
      itr->points.push_back(loc);
    }
  else
    {
      // Need to create a new entry in the table
      table_entry entry;
      entry.bin = this->point_to_bin( loc );
      entry.points.push_back( loc );
      m_table[index].push_back( entry );
      m_num_bin_entries ++ ;

      //  Resize the table right here if needed
      const float resize_multiplier = 1.5;
      if ( m_num_bin_entries > resize_multiplier * m_table.size() )
        {
          std::vector< std::list<table_entry> > new_table( 2*m_table.size() + 1 );
          for ( unsigned int i = 0; i<m_table.size(); ++i )
            for ( table_entry_iterator p = m_table[i].begin(); 
                  p != m_table[i].end(); ++p )
              {
                unsigned k = hash_value( p->bin ) % new_table.size();
                new_table[k].push_back( *p );
              }
          m_table = new_table;
        }
    }
  m_num_points ++ ;
}


void 
geo_hash::add_points( std::vector<point> const& locs )
{
  //  Just repeated apply add_point for an individual point.
  for ( unsigned int i=0; i<locs.size(); ++i )
    this->add_point( locs[i] );
}

//  
inline float square( float a ) { return a*a; }


std::vector<point>
geo_hash::points_in_circle( point center, float radius ) const
{
  // Establish bounds on the bins that could intersect the circle.
  bin_index lower = point_to_bin( point( center.x()-radius, center.y()-radius) );
  bin_index upper = point_to_bin( point( center.x()+radius, center.y()+radius) );

  int   index;
  const_table_entry_iterator itr;
  std::vector<point> points_found;  // record the points found

  //  Check each bin falling within the bounds.
  for ( int i = lower.i(); i<=upper.i(); ++i )
    for ( int j = lower.j(); j<=upper.j(); ++j )
      //  Is the bin occupied?
      if ( find_entry_iterator( bin_index(i,j), index, itr ) )
        {
          //  Check each point
          for ( std::list<point>::const_iterator p = itr->points.begin();
                p != itr->points.end(); ++p )
             { 
               //  If inside the circle, save the point
               if ( square(p->x() - center.x()) + square(p->y() - center.y())
                    <= square(radius) )
                 points_found.push_back( *p );
             }
        }

  std::sort( points_found.begin(), points_found.end() );
  return points_found;
}


std::vector<point>
geo_hash::points_in_rectangle( point min_point, point max_point ) const
{
  //  Establish bounds on the bins
  bin_index lower = point_to_bin( min_point );
  bin_index upper = point_to_bin( max_point );

  int   index;
  const_table_entry_iterator itr;
  std::vector<point> points_found;

  //  Check each bin
  for ( int i = lower.i(); i<=upper.i(); ++i )
    for ( int j = lower.j(); j<=upper.j(); ++j )
      //  Is the bin occupied?
      if ( find_entry_iterator( bin_index(i,j), index, itr ) )
        {
          // Check each point
          for ( std::list<point>::const_iterator p = itr->points.begin();
                p != itr->points.end(); ++p )
             {
               //  If it is actually inside the rectangle then save it
               if ( min_point.x() <= p->x() && p->x() <= max_point.x() &&
                    min_point.y() <= p->y() && p->y() <= max_point.y() )
                 points_found.push_back( *p );
             }
        }
    
  std::sort( points_found.begin(), points_found.end() );
  return points_found;
}


int
geo_hash::erase_points( point center, float radius )
{
  // Find the search range of bins 
  bin_index lower = point_to_bin( point( center.x()-radius, center.y()-radius) );
  bin_index upper = point_to_bin( point( center.x()+radius, center.y()+radius) );

  int   index;
  table_entry_iterator itr;
  
  int num_erased = 0;    // keep track of the number of points erased

  //  For each bin
  for ( int i = lower.i(); i<=upper.i(); ++i )
    for ( int j = lower.j(); j<=upper.j(); ++j )
      //  If the bin is non-empty
      if ( find_entry_iterator( bin_index(i,j), index, itr ) )
        {
          // For  each point in the bin
          for ( std::list<point>::iterator p = itr->points.begin();
                p != itr->points.end(); )
            {
              //  If the point is withn the radius
              if ( square(p->x() - center.x()) + square(p->y() - center.y())
                   <= square(radius) )
                {
                  //  Erase it
                  p = itr->points.erase(p);
                  m_num_points -- ;
                  num_erased ++ ;
                }
              else
                ++p;
            }
          //  Remove the bin if it is empty
          if ( itr->points.empty() )
            {
              m_table[index].erase( itr );
              m_num_bin_entries -- ;
            }
        }

  return num_erased;
}


//  Point location to bin index
bin_index
geo_hash::point_to_bin( point loc ) const
{
  int i = int( floor(loc.x() / m_width) );
  int j = int( floor(loc.y() / m_width) );
  return bin_index(i,j);
}


unsigned int 
geo_hash::hash_value( bin_index bin ) const
{
  return std::abs( bin.i() * 378551 + bin.j() * 63689 );
}


unsigned int 
geo_hash::hash_value( point loc ) const
{
  return hash_value( point_to_bin(loc) );
}


//  Find all points in the bin associated 
std::vector<point>
geo_hash::points_in_bin( point loc ) const
{
  std::vector<point> points_found;
  int   index;
  const_table_entry_iterator itr;

  if ( find_entry_iterator( loc, index, itr ) )  
    {
      points_found.resize( itr->points.size() );
      std::copy( itr->points.begin(), itr->points.end(), points_found.begin() );
      std::sort( points_found.begin(), points_found.end() );
    }
  return points_found;
}


int
geo_hash::num_non_empty() const
{
  return m_num_bin_entries;
}

int
geo_hash::num_points() const
{
  return m_num_points;
}

int
geo_hash::table_size() const
{
  return int( m_table.size() );
}


bool
geo_hash::find_entry_iterator( point                    loc, 
                               int                    & table_index,
                               table_entry_iterator   & itr)
{
  bin_index bin = this->point_to_bin( loc );
  return find_entry_iterator( bin, table_index, itr );
}

bool
geo_hash::find_entry_iterator( point                        loc, 
                               int                        & table_index,
                               const_table_entry_iterator & itr) const
{
  bin_index bin = this->point_to_bin( loc );
  return find_entry_iterator( bin, table_index, itr );
}


bool
geo_hash::find_entry_iterator( bin_index              bin, 
                               int                  & table_index,
                               table_entry_iterator & itr)
{
  table_index = this->hash_value( bin ) % this->table_size();
  for ( itr = m_table[table_index].begin();  
        itr != m_table[table_index].end() && ! (itr->bin == bin); ++itr )
    ;
  return ( itr != m_table[table_index].end() );
}


bool
geo_hash::find_entry_iterator( bin_index                    bin, 
                               int                        & table_index,
                               const_table_entry_iterator & itr) const
{
  table_index = this->hash_value( bin ) % this->table_size();
  for ( itr = m_table[table_index].begin();  
        itr != m_table[table_index].end() && ! (itr->bin == bin); ++itr )
    ;
  return ( itr != m_table[table_index].end() );
}


