#ifndef MC__REVMCCORMICK_H
#define MC__REVMCCORMICK_H 

#include <vector>
#include <iostream>
#include <iomanip>
#include <stdarg.h>
#include <cassert>
#include <string>

namespace mc
{
    //! @brief C++ class for RevMcCormick relaxation arithmetic for factorable function
    ////////////////////////////////////////////////////////////////////////
    //! mc::RevMcCormick is a C++ class computing the McCormick
    //! convex/concave relaxations of factorable functions on a box,
    //! as well as doing reverse subgradient propagation. The template parameter
    //! corresponds to the type used in the underlying interval arithmetic
    //! computations.
    ////////////////////////////////////////////////////////////////////////
    template <typename T>

    class RevMcCormick
    {

        template <typename U> friend class RevMcCormick;

        //! @brief +
        template <typename U> friend RevMcCormick<U> operator+
            (const RevMcCormick<U>&, const RevMcCormick<U>&);

        //! @brief *

        template <typename U> friend RevMcCormick<U> operator*
            (const RevMcCormick<U>&, const RevMcCormick<U>&);
        //template <typename U> friend RevMcCormick<U> operator*
            //(const RevMcCormick<U>&, const double*);
        template <typename U> friend RevMcCormick<U> operator*
            (const RevMcCormick<U>&, const U&);

        //! @brief output
        template <typename U> friend std::ostream& operator<<
            (std::ostream&, const RevMcCormick<U>&);


    public:

        //! @brief =
        RevMcCormick<T>& operator=
        (const RevMcCormick<T>& );

        /** @defgroup MCCORMICK McCormick Relaxation Arithmetic for Factorable Functions
         *  @{
         */
         //! @brief Options of mc::RevMcCormick
        
        static struct Options
        {
            //! @brief Constructor
            Options() :
                DISPLAY_DIGITS(5)
                {}
            //! @brief Number of digits displayed with << operator (default=5)
            unsigned int DISPLAY_DIGITS;
        } options;
        

        //! @brief Exceptions of mc::McCormick
        class Exceptions
        {
        public:
            //! @brief Enumeration type for McCormick exception handling
            enum TYPE {
                DIV = 1,	//!< Division by zero
                INV,	//!< Inverse with zero in range
                LOG,	//!< Log with negative values in range
                SQRT,	//!< Square-root with nonpositive values in range
                ASIN,	//!< Inverse sine or cosine with values outside of \f$[-1,1]\f$ range
                TAN,	//!< Tangent with values outside of \f$[-\frac{\pi}{2}+k\pi,\frac{\pi}{2}+k\pi]\f$ range
                CHEB,	//!< Chebyshev basis function different from [-1,1] range
                MULTSUB = -3,	//!< Failed to propagate subgradients for a product term with Tsoukalas & Mitsos's multivariable composition result
                ENVEL, 	//!< Failed to compute the convex or concave envelope of a univariate term
                SUB	//!< Inconsistent subgradient dimension between two mc::McCormick variables
            };
            //! @brief Constructor for error <a>ierr</a>
            Exceptions(TYPE ierr) : _ierr(ierr) {}
            //! @brief Inline function returning the error flag
            int ierr() { return _ierr; }
            //! @brief Return error description
            std::string what() {
                switch (_ierr) {
                case DIV:
                    return "mc::RevMcCormick\t Division by zero";
                case INV:
                    return "mc::RevMcCormick\t Inverse with zero in range";
                case LOG:
                    return "mc::RevMcCormick\t Log with negative values in range";
                case SQRT:
                    return "mc::RevMcCormick\t Square-root with nonpositive values in range";
                case ASIN:
                    return "mc::RevMcCormick\t Inverse sine with values outside of [-1,1] range";
                case TAN:
                    return "mc::RevMcCormick\t Tangent with values pi/2+k*pi in range";
                case CHEB:
                    return "mc::RevMcCormick\t Chebyshev basis outside of [-1,1] range";
                case MULTSUB:
                    return "mc::RevMcCormick\t Subgradient propagation failed";
                case ENVEL:
                    return "mc::RevMcCormick\t Convex/concave envelope computation failed";
                case SUB:
                    return "mc::RevMcCormick\t Inconsistent subgradient dimension";
                }
                return "mc::RevMcCormick\t Undocumented error";
            }

        private:
            TYPE _ierr;
        };


        //! @brief Default constructor (needed to declare arrays of McCormick class)
        RevMcCormick() :
           _nsubbar(0), _cvsubbar(0), _ccsubbar(0)
        {

        }

        //! @brief Constructor for a constant value <a>c</a>
        RevMcCormick
        (const double c) :
            _nsubbar(0), _cvsubbar(0), _ccsubbar(0)
        {
            _MC = Op<T>::MC(c);
        }

        //! @brief Constructor for an McCmormick MC
        RevMcCormick
        (const T& MC) :
            _nsubbar(0), _cvsubbar(0), _ccsubbar(0)
        {
            _MC = Op<T>::I1(_MC, MC);

        }


        //! @brief Number of reverse version of subgradient components/directions
        unsigned int nsubbar() const
        {
            return _nsubbar;

        }


        //! @brief Number of subgradient components/directions
        unsigned int nsub() const
        {
            return  Op<T>::nsub(_MC);
        }


        //! @brief McCormick relaxations
        T& MC()
        {
            return _MC;
        }
        const T& MC() const
        {
            return _MC;
        }


        //! @brief Lower bound
        double l() const
        {
            return Op<T>::l(_MC);
        }
        //! @brief Upper bound
        double u() const
        {
            return Op<T>::u(_MC);
        }


        //! @brief Convex bound
        double cv() const
        {
            return Op<T>::cv(_MC);
        }
        //! @brief Concave bound
        double cc() const
        {
            return Op<T>::cc(_MC);
        }


        //! @brief <a>i</a>th component of a subgradient of convex underestimator
        double cvsub
        (const unsigned int i) const
        {
            return Op<T>::cvsub(_MC, i);
        }
        //! @brief <a>i</a>th component of a subgradient of concave overestimator
        double ccsub
        (const unsigned int i) const
        {
            return Op<T>::ccsub(_MC, i);
        }


        //! @brief <a>i</a>th component of a reverse subgradient of convex underestimator
        double cvsubbar
        (const unsigned int i) const
        {
            return _cvsubbar[i];
        }
        //! @brief <a>i</a>th component of a reverse subgradient of concave overestimator
        double ccsubbar
        (const unsigned int i) const
        {
            return _ccsubbar[i];
        }


        //! @brief Set dimension of subgradient to <a>nsub</a>
        RevMcCormick<T>& subbar
        (const unsigned int nsubbar);
        //! @brief Set dimension of subgradient to <a>nsub</a> and subgradient values for the convex and concave relaxations to <a>cvsub</a> and <a>ccsub</a>
        RevMcCormick<T>& subbar
        ( const unsigned int nsubbar, const double*cvsubbar, const double*ccsubbar);
      
        //! @brief Set dimension of subgradient to <a>nsub</a> and subgradient values 
        //! for the convex and concave relaxations to <a>cvsub</a> and <a>ccsub</a>              
        
        void sub
        (const unsigned int nsub, const double cvsub[], const double ccsub[])
        {
            Op<T>::Sub(_MC, nsub, cvsub, ccsub);
        }

        
    private:

        //! @brief Number of subgradient components
        unsigned int _nsubbar;
        //! @brief McCormick
        T _MC;
        //! @brief Reverse subgradient of convex underestimator
        double *_cvsubbar;
        //std::vector<double> _cvsubbar;
        //! @brief Reverse subgradient of concave overestimator
        //std::vector<double> _ccsubbar;
        double *_ccsubbar;
        //! @brief Copy subgradient arrays
        void _subbar_copy(const RevMcCormick<T>& RevMC);
        //! @brief Initialize subgradient arrays
        void _subbar(const unsigned int nsubbar);
        void _subbar_resize( const unsigned int nsubbar);
        RevMcCormick<T>& _sum( const RevMcCormick<T>&RevMC1, const RevMcCormick<T>&RevMC2 );
        RevMcCormick<T>& _mul1( const RevMcCormick<T>&RevMC1, const RevMcCormick<T>&RevMC2 );
        //RevMcCormick<T>& _mul2( const RevMcCormick<T>&RevMC1, const double* tempsub);

    };


    ////////////////////////////////////////////////////////////////////////
    template <typename T> inline void
    RevMcCormick<T>::_subbar_copy
    (const RevMcCormick<T>& RevMC)
    {
        _subbar_resize( RevMC._nsubbar);
        for ( unsigned int i=0; i<_nsubbar; i++ ){
        _cvsubbar[i] = RevMC._cvsubbar[i];
        _ccsubbar[i] = RevMC._ccsubbar[i];
        }
        return;
    }



    template <typename T> inline void
        RevMcCormick<T>::_subbar_resize
    ( const unsigned int nsubbar )
    {
        _nsubbar = nsubbar;
        _cvsubbar = new double[_nsubbar];
        _ccsubbar = new double[_nsubbar];
    }



    template <typename T> inline void
        RevMcCormick<T>::_subbar
        (const unsigned int nsubbar)
    {
        _subbar_resize( nsubbar );
        for ( unsigned int i=0; i<nsubbar; i++ ){
            _cvsubbar[i] = _ccsubbar[i] = 0.;
        }
    }


    template <typename T> inline RevMcCormick<T>&
    RevMcCormick<T>::subbar
    ( const unsigned int nsubbar)
    {
        _subbar(nsubbar);
        return *this;
    }


    template <typename T> inline RevMcCormick<T>&
    RevMcCormick<T>::subbar
    ( const unsigned int nsubbar, const double*cvsubbar, const double*ccsubbar)
    {
        subbar( nsubbar );
        for ( unsigned int i=0; i<nsubbar; i++ ){
        _cvsubbar[i] = cvsubbar[i];
        _ccsubbar[i] = ccsubbar[i];
        }
        return *this;
    }


    ////////////////////////////////////////////////////////////////////////
    template <typename T> inline RevMcCormick<T>&
        RevMcCormick<T>::operator=
        (const RevMcCormick<T>& RevMC)
    {
        _MC = RevMC._MC;
        _subbar_copy(RevMC);
        return *this;
    }


    template <typename T> inline RevMcCormick<T>&
    RevMcCormick<T>::_sum
    ( const RevMcCormick<T>&RevMC1, const RevMcCormick<T>&RevMC2 )
    {
        //_MC = RevMC1._MC;
        for (unsigned int i = 0; i < _nsubbar; i++) {
            _cvsubbar[i] = RevMC1._cvsubbar[i] + RevMC2._cvsubbar[i];
            _ccsubbar[i] = RevMC1._ccsubbar[i] + RevMC2._ccsubbar[i];
        }
        return *this;
    }


    template <typename T> inline RevMcCormick<T>&
    RevMcCormick<T>::_mul1
    ( const RevMcCormick<T>&RevMC1, const RevMcCormick<T>&RevMC2 )
    {        
        _cvsubbar[0] = RevMC1._cvsubbar[0] * RevMC2.cvsub(0)
            + RevMC1._cvsubbar[1] * RevMC2.ccsub(0);

        _cvsubbar[1] = RevMC1._cvsubbar[1] * RevMC2.ccsub(1)
            + RevMC1._cvsubbar[0] * RevMC2.cvsub(1);

        _ccsubbar[0] = RevMC1._ccsubbar[0] * RevMC2.cvsub(0)
            + RevMC1._ccsubbar[1] * RevMC2.ccsub(0);

        _ccsubbar[1] = RevMC1._ccsubbar[1] * RevMC2.ccsub(1)
            + RevMC1._ccsubbar[0] * RevMC2.cvsub(1);
        return *this;
    }

    
    template <typename T> inline RevMcCormick<T>
    operator+

        (const RevMcCormick<T>& RevMC1, const RevMcCormick<T>& RevMC2)
    {

        RevMcCormick<T> RevMC3;
        RevMC3._subbar(RevMC1._nsubbar);

        return  RevMC3._sum(RevMC1,RevMC2);
        
    }



    template <typename T> inline RevMcCormick<T>
    operator*
        ( const RevMcCormick<T>& RevMC1, const RevMcCormick<T>& RevMC2)
    {
        RevMcCormick<T> RevMC3; 
        RevMC3._subbar(RevMC1._nsubbar);

        return RevMC3._mul1(RevMC1, RevMC2);
    }

    template <typename T> inline RevMcCormick<T>
    operator*
        ( const RevMcCormick<T>& RevMC1, const T& MC1)
    {
        RevMcCormick<T> RevMC3; 
        RevMC3._subbar(RevMC1._nsubbar);

        RevMC3._cvsubbar[0] = RevMC1._cvsubbar[0] * MC1.cvsub(0)
            + RevMC1._cvsubbar[1] * MC1.ccsub(0);

        RevMC3._cvsubbar[1] = RevMC1._cvsubbar[1] * MC1.ccsub(1)
            + RevMC1._cvsubbar[0] * MC1.cvsub(1);

        RevMC3._ccsubbar[0] = RevMC1._ccsubbar[0] * MC1.cvsub(0)
            + RevMC1._ccsubbar[1] * MC1.ccsub(0);

        RevMC3._ccsubbar[1] = RevMC1._ccsubbar[1] * MC1.ccsub(1)
            + RevMC1._ccsubbar[0] * MC1.cvsub(1);

        return RevMC3;
    }

    ////////////////////////////////////////////////////////////////////////

    template <typename T> inline std::ostream&
        operator<<
        (std::ostream& out, const RevMcCormick<T>& RevMC)
    {

        out << std::scientific << std::setprecision(RevMcCormick<T>::options.DISPLAY_DIGITS) << std::right
            << "[ " << std::setw(RevMcCormick<T>::options.DISPLAY_DIGITS + 7) << RevMC.l() << " : "
            << std::setw(RevMcCormick<T>::options.DISPLAY_DIGITS + 7) << RevMC.u()
            << " ] [ " << std::setw(RevMcCormick<T>::options.DISPLAY_DIGITS + 7) << RevMC.cv() << " : "
            << std::setw(RevMcCormick<T>::options.DISPLAY_DIGITS + 7) << RevMC.cc() << " ]";
        
        if (RevMC.nsub()) {
            out << " [ (";
            for (unsigned int i = 0; i < RevMC.nsub() - 1; i++)
                out << std::setw(RevMcCormick<T>::options.DISPLAY_DIGITS + 7) << RevMC.cvsub(i) << ",";
            out << std::setw(RevMcCormick<T>::options.DISPLAY_DIGITS + 7) << RevMC.cvsub(RevMC.nsub() - 1) << ") : (";
            for (unsigned int i = 0; i < RevMC.nsub() - 1; i++)
                out << std::setw(RevMcCormick<T>::options.DISPLAY_DIGITS + 7) << RevMC.ccsub(i) << ",";
            out << std::setw(RevMcCormick<T>::options.DISPLAY_DIGITS + 7) << RevMC.ccsub(RevMC.nsub() - 1) << ") ]";
        }

        
        if (RevMC.nsubbar()) {
            out << " [ (";
            for (unsigned int i = 0; i < RevMC.nsubbar() - 1; i++)
                out << std::setw(RevMcCormick<T>::options.DISPLAY_DIGITS + 7) << RevMC.cvsubbar(i) << ",";
            out << std::setw(RevMcCormick<T>::options.DISPLAY_DIGITS + 7) << RevMC.cvsubbar(RevMC.nsubbar() - 1) << ") : (";
            for (unsigned int i = 0; i < RevMC.nsubbar() - 1; i++)
                out << std::setw(RevMcCormick<T>::options.DISPLAY_DIGITS + 7) << RevMC.ccsubbar(i) << ",";
            out << std::setw(RevMcCormick<T>::options.DISPLAY_DIGITS + 7) << RevMC.ccsubbar(RevMC.nsubbar() - 1) << ") ]";
        }
        
        return out;
    }

    template <typename T> typename RevMcCormick<T>::Options RevMcCormick<T>::options;

} // namespace mc


#endif
